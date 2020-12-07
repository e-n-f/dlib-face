// Following:
//
// http://dlib.net/dnn_face_recognition_ex.cpp.html
// https://github.com/ageitgey/face_recognition/blob/master/face_recognition/api.py

// for vasprintf() on Linux
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <getopt.h>
#include <pthread.h>
#include <stdarg.h>
#include <unistd.h>

using namespace dlib;

size_t triangles[][3] = {
	#include "triangulation.h"
};
size_t ntriangles = (sizeof(triangles) / (3 * sizeof(size_t)));

bool flop = false;
size_t nth = 0;
double contrast = 1;
bool half = false;

struct mean_stddev {
	size_t count = 0;
	double themean = 0;
	double m2 = 0;

	void add(double diff) {
		count++;
		double delta = diff - themean;
		themean = themean + delta / count;
		double delta2 = diff - themean;
		m2 = m2 + delta * delta2;
	}

	double stddev() {
		return sqrt(m2 / count);
	}

	double mean() {
		return themean;
	}

	void set_mean_stddev(double mean, double stddev) {
		themean = mean;
		count = 1;
		m2 = stddev * stddev;
	}
};

struct face {
	size_t seq;
	std::string bbox;
	std::vector<std::string> landmarks;
	std::vector<float> metrics;
	std::string fname;
};

std::vector<face> faces;

std::string nextline(FILE *f) {
	std::string out;

	int c;
	while ((c = getc(f)) != EOF) {
		out.push_back(c);

		if (c == '\n') {
			break;
		}
	}

	return out;
}

std::string gettok(std::string &s) {
	std::string out;

	while (s.size() > 0 && s[0] != ' ') {
		out.push_back(s[0]);
		s.erase(s.begin());
	}

	if (s.size() > 0 && s[0] == ' ') {
		s.erase(s.begin());
	}

	return out;
}

face toface(std::string s) {
	std::string tok;
	face f;

	tok = gettok(s);
	f.seq = atoi(tok.c_str());

	tok = gettok(s);
	f.bbox = tok;

	while (true) {
		tok = gettok(s);
		if (tok == "--") {
			break;
		}
		f.landmarks.push_back(tok);
	}

	for (size_t i = 0; i < 128; i++) {
		tok = gettok(s);
		f.metrics.push_back(atof(tok.c_str()));
	}

	f.fname = s;
	return f;
}

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, and the training dataset consisted of about 3 million images instead of
// 55.  Also, the input layer was locked to images of size 150.
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

void aprintf(std::string &buf, const char *format, ...) {
        va_list ap;
        char *tmp;

        va_start(ap, format);
        if (vasprintf(&tmp, format, ap) < 0) {
                fprintf(stderr, "memory allocation failure\n");
                exit(EXIT_FAILURE);
        }
        va_end(ap);

        buf.append(tmp, strlen(tmp));
        free(tmp);
}

struct state {
	std::vector<std::string> fnames;
	dlib::frontal_face_detector detector;
	dlib::shape_predictor sp;
	anet_type net;
};

struct arg {
	std::vector<std::string> *fnames;
	dlib::frontal_face_detector *detector;
	dlib::shape_predictor *sp;
	anet_type *net;
};

void maptri(matrix<rgb_pixel> &img_in, full_object_detection &landmarks_in,
            matrix<rgb_pixel> &img_out, full_object_detection &landmarks_out,
	    size_t triangle[3],
	    std::vector<mean_stddev> &histogram_in,
	    std::vector<mean_stddev> &histogram_out,
	    bool pass, matrix<rgb_pixel> &already_in, matrix<rgb_pixel> &already_out) {
	long x0_in = landmarks_in.part(triangle[0])(0);
	long y0_in = landmarks_in.part(triangle[0])(1);

	long x1_in = landmarks_in.part(triangle[1])(0);
	long y1_in = landmarks_in.part(triangle[1])(1);

	long x2_in = landmarks_in.part(triangle[2])(0);
	long y2_in = landmarks_in.part(triangle[2])(1);

	long x0_out = landmarks_out.part(triangle[0])(0);
	long y0_out = landmarks_out.part(triangle[0])(1);

	long x1_out = landmarks_out.part(triangle[1])(0);
	long y1_out = landmarks_out.part(triangle[1])(1);

	long x2_out = landmarks_out.part(triangle[2])(0);
	long y2_out = landmarks_out.part(triangle[2])(1);

	double d01 = sqrt((x1_out - x0_out) * (x1_out - x0_out) + (y1_out - y0_out) * (y1_out - y0_out));
	double d12 = sqrt((x2_out - x1_out) * (x2_out - x1_out) + (y2_out - y1_out) * (y2_out - y1_out));
	double d20 = sqrt((x0_out - x2_out) * (x0_out - x2_out) + (y0_out - y2_out) * (y0_out - y2_out));

	double longest = std::max(d01, std::max(d12, d20));

	for (size_t i = 0; i < std::ceil(longest); i++) {
		double x01_in = x0_in + (x1_in - x0_in) / longest * i;
		double y01_in = y0_in + (y1_in - y0_in) / longest * i;

		double x02_in = x0_in + (x2_in - x0_in) / longest * i;
		double y02_in = y0_in + (y2_in - y0_in) / longest * i;

		double x01_out = x0_out + (x1_out - x0_out) / longest * i;
		double y01_out = y0_out + (y1_out - y0_out) / longest * i;

		double x02_out = x0_out + (x2_out - x0_out) / longest * i;
		double y02_out = y0_out + (y2_out - y0_out) / longest * i;

		for (size_t j = 0; j < std::ceil(longest); j++) {
			long x_in = x01_in + (x02_in - x01_in) / longest * j;
			long y_in = y01_in + (y02_in - y01_in) / longest * j;

			long x_out = x01_out + (x02_out - x01_out) / longest * j;
			long y_out = y01_out + (y02_out - y01_out) / longest * j;

			if (x_out >= 0 && x_out < img_out.nc() &&
			    y_out >= 0 && y_out < img_out.nr() &&
			    x_in >= 0 && x_in < img_in.nc() &&
			    y_in >= 0 && y_in < img_in.nr()) {
				if (pass) {
					double red = (img_in(y_in, x_in).red - histogram_in[0].mean()) / histogram_in[0].stddev() * contrast * histogram_out[0].stddev() + histogram_out[0].mean();
					double green = (img_in(y_in, x_in).green - histogram_in[1].mean()) / histogram_in[1].stddev() * contrast * histogram_out[1].stddev() + histogram_out[1].mean();
					double blue = (img_in(y_in, x_in).blue - histogram_in[2].mean()) / histogram_in[2].stddev() * contrast * histogram_out[2].stddev() + histogram_out[2].mean();

					if (red < 0) {
						red = 0;
					}
					if (green < 0) {
						green = 0;
					}
					if (blue < 0) {
						blue = 0;
					}
					if (red > 255) {
						red = 255;
					}
					if (green > 255) {
						green = 255;
					}
					if (blue > 255) {
						blue = 255;
					}

					rgb_pixel rgb;
					rgb.red = red;
					rgb.green = green;
					rgb.blue = blue;

					img_out(y_out, x_out) = rgb;
				} else {
					if (!already_in(y_in, x_in).red) {
						histogram_in[0].add(img_in(y_in, x_in).red);
						histogram_in[1].add(img_in(y_in, x_in).green);
						histogram_in[2].add(img_in(y_in, x_in).blue);
						already_in(y_in, x_in).red = 1;
					}

					if (!already_out(y_out, x_out).green) {
						histogram_out[0].add(img_out(y_out, x_out).red);
						histogram_out[1].add(img_out(y_out, x_out).green);
						histogram_out[2].add(img_out(y_out, x_out).blue);
						already_out(y_out, x_out).green = 1;
					}
				}
			}
		}
	}
}

double dist(double x1, double y1, double x2, double y2) {
	double xd = x1 - x2;
	double yd = y1 - y2;
	return sqrt(xd * xd + yd * yd);
}

std::vector<std::pair<double, double>> face2double(face &f) {
	std::vector<std::pair<double, double>> ret;
	for (size_t i = 0; i < f.landmarks.size(); i++) {
		double x, y;
		sscanf(f.landmarks[i].c_str(), "%lf,%lf", &x, &y);
		ret.push_back(std::pair<double, double>(x, y));
	}

	if (ret.size() != 68) {
		return ret;
	}

	double nose_x = ret[27].first;
	double nose_y = ret[27].second;

	double chin_x = ret[8].first;
	double chin_y = ret[8].second;

	double xd = nose_x - chin_x;
	double yd = nose_y - chin_y;
	double angle = atan2(yd, xd);
	double dist = sqrt(xd * xd + yd * yd);

	for (size_t i = 0; i < ret.size(); i++) {
		double xxd = ret[i].first - nose_x;
		double yyd = ret[i].second - nose_y;
		double ang = atan2(yyd, xxd);
		double d = sqrt(xxd * xxd + yyd * yyd);
		ang -= angle - M_PI / 2;
		ret[i].first = d / dist * cos(ang);
		ret[i].second = d / dist * sin(ang);
	}

	return ret;
}

double calc_landmark_similarity(face &a, face &b) {
	std::vector<std::pair<double, double>> a1 = face2double(a);
	std::vector<std::pair<double, double>> b1 = face2double(b);

	if (a1.size() != 68 || b1.size() != 68) {
		fprintf(stderr, "can't compare %zu and %zu landmark faces\n", a1.size(), b1.size());
		return 999;
	}

	double badness = 0;

	for (size_t i = 0; i < a1.size(); i++) {
		double xd = a1[i].first - b1[i].first;
		double yd = a1[i].second - b1[i].second;
		double d = xd * xd + yd * yd;
		badness += d;
	}

	return sqrt(badness);
}

face find_best(full_object_detection &landmarks) {
	face landmarks_face;

	landmarks_face.landmarks.resize(landmarks.num_parts());
	for (size_t i = 0; i < landmarks_face.landmarks.size(); i++) {
		landmarks_face.landmarks[i].append(std::to_string(landmarks.part(i)(0)) + std::string(",") + std::to_string(landmarks.part(i)(1)));
	}

	std::map<double, face> options;

	for (size_t i = 0; i < faces.size(); i++) {
		if (faces[i].landmarks.size() != 68) {
			fprintf(stderr, "Expected 68 landmarks, not %zu, for %s\n", faces[i].landmarks.size(), faces[i].fname.c_str());
			exit(EXIT_FAILURE);
		}

		double badness = calc_landmark_similarity(faces[i], landmarks_face);
		options.insert(std::pair<double, face>(badness, faces[i]));

		// printf("%.12f %s\n", badness, faces[i].fname.c_str());
	}

	auto n = options.begin();
	std::advance(n, nth);
	return n->second;
}

matrix<rgb_pixel> clearpix(matrix<rgb_pixel> const &p) {
	matrix<rgb_pixel> ret = p;
	for (size_t x = 0; x < ret.nc(); x++) {
		for (size_t y = 0; y < ret.nr(); y++) {
			ret(y, x).red = 0;
			ret(y, x).green = 0;
			ret(y, x).blue = 0;
		}
	}
	return ret;
}

void *run1(void *v) {
	arg *a = (arg *) v;

	std::vector<std::string> *fnames = a->fnames;
	dlib::frontal_face_detector *detector = a->detector;
	dlib::shape_predictor *sp = a->sp;
	anet_type *net = a->net;

	std::string ret;

	std::vector<matrix<rgb_pixel>> imgs;
	std::vector<full_object_detection> landmarkses;

	for (size_t a = 0; a < fnames->size(); a++) {
		std::string fname = (*fnames)[a];
		matrix<rgb_pixel> img;

		if (fname.size() > 0 && fname[0] != '/') {
			static bool warned = false;
			if (!warned) {
				fprintf(stderr, "Warning: %s is not an absolute path\n", fname.c_str());
				warned = true;
			}
		}

		try {
			load_image(img, fname);
		} catch (...) {
			fprintf(stderr, "%s: failed image loading\n", fname.c_str());
			continue;
		}

		double scale = 1;

		while (img.size() > 4000 * 3000 * sqrt(2)) {
			// printf("scale down: %ldx%ld\n", img.nc(), img.nr());
			pyramid_down<2> pyr;
			matrix<rgb_pixel> tmp;
			pyr(img, tmp);
			img = tmp;
			scale /= 2;
		}
		while (img.size() < 1024 * 750 / sqrt(2)) {
			// printf("scale up: %ldx%ld\n", img.nc(), img.nr());
			pyramid_up(img);
			scale *= 2;
		}

		std::vector<matrix<rgb_pixel>> faces;
		std::vector<full_object_detection> landmarks;

		for (auto face : (*detector)(img)) {
			full_object_detection shape = (*sp)(img, face);
			landmarks.push_back(shape);

			matrix<rgb_pixel> face_chip;
			extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);

			if (flop) {
				matrix<rgb_pixel> face_chip2 = face_chip;

				for (size_t x = 0; x < face_chip.nc(); x++) {
					for (size_t y = 0; y < face_chip.nr(); y++) {
						face_chip2(y, x) = face_chip(y, face_chip.nc() - 1 - x);
					}
				}

				face_chip = face_chip2;
			}

			faces.push_back(std::move(face_chip));
		}

		std::vector<matrix<float, 0, 1>> face_descriptors = (*net)(faces);

		if (faces.size() != landmarks.size()) {
			aprintf(ret, "%zu faces but %zu landmarks\n", faces.size(), landmarks.size());
			continue;
		}

		if (faces.size() == 0) {
		}

		for (size_t i = 0; i < face_descriptors.size(); i++) {
			// rectangle rect = landmarks[i].get_rect();

			// long width = rect.right() - rect.left();
			// long height = rect.bottom() - rect.top();

			for (size_t j = 0; j < landmarks[i].num_parts(); j++) {
				point p = landmarks[i].part(j);
#if 0
				// Not scaling because we are pushing the scaled version of the image
				// for triangle swapping, not the original
				landmarks[i].part(j)(0) = p(0) / scale;
				landmarks[i].part(j)(1) = p(1) / scale;
#endif
			}

			if (fnames->size() > 1 || imgs.size() == 0) {
				imgs.push_back(img);
			}

			landmarkses.push_back(landmarks[i]);
		}
	}

	std::vector<std::vector<mean_stddev>> histograms_in, histograms_out;
	histograms_in.resize(landmarkses.size());
	histograms_out.resize(landmarkses.size());

	for (size_t i = 0; i < landmarkses.size(); i++) {
		histograms_in[i].resize(3);
		histograms_out[i].resize(3);
	}

	if (imgs.size() == 1) {
		matrix<rgb_pixel> out = imgs[0];

		if (faces.size() == 0) {
			fprintf(stderr, "Removed these other cases\n");
			exit(EXIT_FAILURE);
		} else {
			std::vector<full_object_detection> landmarks_in;
			std::vector<matrix<rgb_pixel>> imgs_in;

			for (size_t i = 0; i < landmarkses.size(); i++) {
				face best = find_best(landmarkses[i]);

				matrix<rgb_pixel> img;
				try {
					printf("chose %s\n", best.fname.c_str());
					load_image(img, best.fname);
				} catch (...) {
					fprintf(stderr, "%s: failed image loading\n", best.fname.c_str());
					exit(EXIT_FAILURE);
				}

				imgs_in.push_back(img);

				int x, y, w, h;
				if (sscanf(best.bbox.c_str(), "%dx%d+%d+%d", &w, &h, &x, &y) == 4) {
					rectangle face(x, y, (x + w), (y + h));

					full_object_detection shape = (*sp)(img, face);
					landmarks_in.push_back(shape);
				} else {
					fprintf(stderr, "bad bbox %s\n", best.bbox.c_str());
					exit(EXIT_FAILURE);
				}
			}

			for (size_t i = 0; i < landmarkses.size(); i++) {
				matrix<rgb_pixel> already_in = clearpix(imgs_in[i]);
				matrix<rgb_pixel> already_out = clearpix(out);

				// Normalize lightnesses along the forehead transition

				int x1 = (landmarks_in[i].part(19)(0) + landmarks_in[i].part(24)(0)) / 2;
				int y1 = (landmarks_in[i].part(19)(1) + landmarks_in[i].part(24)(1)) / 2;

				int x2 = (landmarkses[i].part(19)(0) + landmarkses[i].part(24)(0)) / 2;
				int y2 = (landmarkses[i].part(19)(1) + landmarkses[i].part(24)(1)) / 2;

				double r1b = imgs_in[i](y1, x1).red;
				double g1b = imgs_in[i](y1, x1).green;
				double b1b = imgs_in[i](y1, x1).blue;

				double r1s = out(y2, x2).red;
				double g1s = out(y2, x2).green;
				double b1s = out(y2, x2).blue;

				for (size_t k = 0; k < ntriangles; k++) {
					// Only do mean/stddev calculation for cheecks and chin
					if ((triangles[k][0] <= 13 && triangles[k][0] >= 03) ||
					    (triangles[k][1] <= 13 && triangles[k][1] >= 03) ||
					    (triangles[k][2] <= 13 && triangles[k][2] >= 03)) {
						maptri(imgs_in[i], landmarks_in[i], out, landmarkses[i], triangles[k], histograms_in[i], histograms_out[i], false, already_in, already_out);
					}
				}

				histograms_in[i][0].set_mean_stddev(r1b, 1);
				histograms_in[i][1].set_mean_stddev(g1b, 1);
				histograms_in[i][2].set_mean_stddev(b1b, 1);

				histograms_out[i][0].set_mean_stddev(r1s, 1);
				histograms_out[i][1].set_mean_stddev(g1s, 1);
				histograms_out[i][2].set_mean_stddev(b1s, 1);

				printf("using %f,%f to %f,%f red\n",
					histograms_in[i][0].mean(), histograms_in[i][0].stddev(),
					histograms_out[i][0].mean(), histograms_out[i][0].stddev());
				printf("using %f,%f to %f,%f green\n",
					histograms_in[i][1].mean(), histograms_in[i][1].stddev(),
					histograms_out[i][1].mean(), histograms_out[i][1].stddev());
				printf("using %f,%f to %f,%f blue\n",
					histograms_in[i][2].mean(), histograms_in[i][2].stddev(),
					histograms_out[i][2].mean(), histograms_out[i][2].stddev());

				for (size_t k = 0; k < ntriangles; k++) {
					maptri(imgs_in[i], landmarks_in[i], out, landmarkses[i], triangles[k], histograms_in[i], histograms_out[i], true, already_in, already_out);
				}
			}
		}

		if (half) {
			matrix<rgb_pixel> out2 = out;

			for (size_t y = 0; y < out.nr(); y++) {
				for (size_t x = 0; x < out.nc(); x++) {
					out2(y, x).red = out(y, x).red / 2 + imgs[0](y, x).red / 2;
					out2(y, x).green = out(y, x).green / 2 + imgs[0](y, x).green / 2;
					out2(y, x).blue = out(y, x).blue / 2 + imgs[0](y, x).blue / 2;
				}
			}

			out = out2;
		}

		char buf[600];
		sprintf(buf, "out-%zu.jpg", (size_t) 0);
		save_jpeg(out, buf);
	} else {
		fprintf(stderr, "Removed these other cases\n");
		exit(EXIT_FAILURE);
	}

	std::string *out = new std::string;
	out->append(ret);
	return (void *) out;
}

void runq(std::vector<arg> &queue) {
	size_t jobs = queue.size();

	std::vector<pthread_t> awaiting;
	awaiting.resize(jobs);

	for (size_t i = 0; i < jobs; i++) {
		if (pthread_create(&awaiting[i], NULL, run1, &queue[i]) < 0) {
			fprintf(stderr, "pthread_create: %s\n", strerror(errno));
			exit(EXIT_FAILURE);
		}
	}

	for (size_t i = 0; i < jobs; i++) {
		void *ret;
		if (pthread_join(awaiting[i], &ret) != 0) {
			fprintf(stderr, "pthread_join: %s\n", strerror(errno));
			exit(EXIT_FAILURE);
		}

		std::string *s = (std::string *) ret;
		if (printf("%s", s->c_str()) < 0) {
			perror("printf");
			exit(EXIT_FAILURE);
		}
		if (ferror(stdout)) {
			perror("stdout");
			exit(EXIT_FAILURE);
		}
		delete(s);
		fflush(stdout);

		queue[i].fnames->clear();
	}
}

void read_candidates(const char *fname) {
	FILE *f = fopen(fname, "r");
	if (f == NULL) {
		perror(fname);
		exit(EXIT_FAILURE);
	}

	std::string s;
	while ((s = nextline(f)).size() != 0) {
		if (s[0] == '#') {
			continue;
		}
		s.resize(s.size() - 1);
		faces.push_back(toface(s));
	}
	fclose(f);
}

void usage(const char *s) {
	fprintf(stderr, "Usage: %s [-j threads]\n", s);
}

int main(int argc, char **argv) {
	size_t jobs = 1;

	int o;
	extern int optind;
	extern char *optarg;

	while ((o = getopt(argc, argv, "fp:n:c:h")) != -1) {
		switch (o) {
		case 'f':
			flop = true;
			break;

		case 'c':
			contrast = atof(optarg);
			break;

		case 'p':
			read_candidates(optarg);
			break;

		case 'n':
			nth = atoi(optarg);
			break;

		case 'h':
			half = true;
			break;

		default:
			usage(*argv);
			exit(EXIT_FAILURE);
		}
	}

	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

	dlib::shape_predictor sp;
	dlib::deserialize("/usr/local/share/shape_predictor_68_face_landmarks.dat") >> sp;

	anet_type net;
	dlib::deserialize("/usr/local/share/dlib_face_recognition_resnet_model_v1.dat") >> net;

	std::vector<state> states;
	states.resize(jobs);

	std::vector<arg> jobq;
	jobq.resize(jobs);

	for (size_t i = 0; i < jobs; i++) {
		states[i].detector = detector;
		states[i].sp = sp;
		states[i].net = net;

		jobq[i].detector = &states[i].detector;
		jobq[i].sp = &states[i].sp;
		jobq[i].net = &states[i].net;
		jobq[i].fnames = &states[i].fnames;
	}

	size_t seq = 0;

	if (optind >= argc) {
		if (isatty(0)) {
			fprintf(stderr, "Warning: standard input is a terminal\n");
		}

		while (true) {
			std::string fname = nextline(stdin);
			seq++;

			if (fname.size() == 0) {
				break;
			}

			fname.resize(fname.size() - 1);
			jobq[seq % jobs].fnames->push_back(fname);

			if (jobq[seq % jobs].fnames->size() >= 20) {
				runq(jobq);
			}
		}
	} else {
		for (; optind < argc; optind++) {
			jobq[optind % jobs].fnames->push_back(argv[optind]);

			if (jobq[optind % jobs].fnames->size() >= 20) {
				runq(jobq);
			}
		}
	}

	runq(jobq);
}
