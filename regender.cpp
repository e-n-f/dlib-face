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

double landmark_pixels[][2] = {
	#include "landmarks-68.h"
};

bool flop = false;
bool landmarks = true;
bool reencode = false;
bool check_reencode = false;

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
};

struct face {
        size_t seq;
        std::string bbox;
        std::vector<std::string> landmarks;
        std::vector<float> metrics;
        std::string fname;

	double distance(face const &f) {
		double diff = 0;
		for (size_t i = 0; i < metrics.size() && i < f.metrics.size(); i++) {
			diff += (metrics[i] - f.metrics[i]) * (metrics[i] - f.metrics[i]);
		}
		diff = sqrt(diff);
		return diff;
	}
};

std::string nextline() {
	std::string out;

	int c;
	while ((c = getchar()) != EOF) {
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
	    bool pass) {
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
					double red = (img_in(y_in, x_in).red - histogram_in[0].mean()) / histogram_in[0].stddev() * histogram_out[0].stddev() + histogram_out[0].mean();
					double green = (img_in(y_in, x_in).green - histogram_in[1].mean()) / histogram_in[1].stddev() * histogram_out[1].stddev() + histogram_out[1].mean();
					double blue = (img_in(y_in, x_in).blue - histogram_in[2].mean()) / histogram_in[2].stddev() * histogram_out[2].stddev() + histogram_out[2].mean();

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
					histogram_in[0].add(img_in(y_in, x_in).red);
					histogram_in[1].add(img_in(y_in, x_in).green);
					histogram_in[2].add(img_in(y_in, x_in).blue);

					histogram_out[0].add(img_out(y_out, x_out).red);
					histogram_out[1].add(img_out(y_out, x_out).green);
					histogram_out[2].add(img_out(y_out, x_out).blue);
				}
			}
		}
	}
}

void *run1(void *v) {
	arg *a = (arg *) v;

	std::vector<std::string> *fnames = a->fnames;
	dlib::frontal_face_detector *detector = a->detector;
	dlib::shape_predictor *sp = a->sp;
	anet_type *net = a->net;

	std::string ret;

	matrix<rgb_pixel> brothers;
	matrix<rgb_pixel> sisters;

	rectangle std_rect(0, 0, 612, 612);
	std::vector<point> std_landmarks;
	for (size_t i = 0; i < 68; i++) {
		point p(landmark_pixels[i][0], landmark_pixels[i][1]);
		std_landmarks.push_back(p);
	}

	full_object_detection standard_landmarks(std_rect, std_landmarks);

	try {
		load_image(brothers, "siblings/brothers.jpg");
	} catch (...) {
		fprintf(stderr, "brothers: failed image loading\n");
		exit(EXIT_FAILURE);
	}

	try {
		load_image(sisters, "siblings/sisters.jpg");
	} catch (...) {
		fprintf(stderr, "sisters: failed image loading\n");
		exit(EXIT_FAILURE);
	}

	for (size_t a = 0; a < fnames->size(); a++) {
		face f;
		std::string fname = (*fnames)[a];
		matrix<rgb_pixel> img;

		if (reencode) {
			if (fname[0] == '#') {
				continue;
			}

			f = toface(fname);
			fname = f.fname;
		}

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

		if (flop) {
			matrix<rgb_pixel> img2 = img;

			for (size_t x = 0; x < img.nc(); x++) {
				for (size_t y = 0; y < img.nr(); y++) {
					img2(y, x) = img(y, img.nc() - 1 - x);
				}
			}

			img = img2;
		}

		std::vector<matrix<rgb_pixel>> faces;
		std::vector<full_object_detection> landmarks;

		if (reencode) {
			int x, y, w, h;
			if (sscanf(f.bbox.c_str(), "%dx%d+%d+%d", &w, &h, &x, &y) == 4) {
				rectangle face(x * scale, y * scale, (x + w) * scale, (y + h) * scale);

				full_object_detection shape = (*sp)(img, face);
				landmarks.push_back(shape);

				matrix<rgb_pixel> face_chip;
				extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);

				faces.push_back(std::move(face_chip));
			} else {
				fprintf(stderr, "Can't parse bounding box %s for %s\n", f.bbox.c_str(), fname.c_str());
			}
		} else {
			for (auto face : (*detector)(img)) {
				full_object_detection shape = (*sp)(img, face);
				landmarks.push_back(shape);

				matrix<rgb_pixel> face_chip;
				extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);

				faces.push_back(std::move(face_chip));
			}
		}

		std::vector<matrix<float, 0, 1>> face_descriptors = (*net)(faces);

		if (faces.size() != landmarks.size()) {
			aprintf(ret, "%zu faces but %zu landmarks\n", faces.size(), landmarks.size());
			continue;
		}

		if (faces.size() == 0) {
			aprintf(ret, "# %s\n", fname.c_str());
		}

		for (size_t i = 0; i < face_descriptors.size(); i++) {
			// The idea here is:
			// * Copy the provided image to two new images
			// * Copy the brother and sister reference faces
			//   to the dimensions of the provided face
			//   in those images
			// * Multiply/divide by the brother/sister faces
			//   (which will be a no-op outside the bounds)
			// * Future: reproportion the resulting face
			// * Use the result of that as the new base image
			//
			// The reason to copy the brothers and sisters
			// instead of doing the transformation per triangle
			// is that as currently implemented the triangle edges
			// would get transformed repeatedly instead of just once

			matrix<rgb_pixel> scaled_brothers = img;
			matrix<rgb_pixel> scaled_sisters = img;
			std::vector<mean_stddev> brothers_in, brothers_out;
			std::vector<mean_stddev> sisters_in, sisters_out;
			brothers_in.resize(3);
			brothers_out.resize(3);
			sisters_in.resize(3);
			sisters_out.resize(3);

			for (size_t k = 0; k < ntriangles; k++) {
				maptri(brothers, standard_landmarks, scaled_brothers, landmarks[i], triangles[k], brothers_in, brothers_out, false);
				maptri(sisters, standard_landmarks, scaled_sisters, landmarks[i], triangles[k], sisters_in, sisters_out, false);
			}

			for (size_t k = 0; k < ntriangles; k++) {
				maptri(brothers, standard_landmarks, scaled_brothers, landmarks[i], triangles[k], brothers_in, brothers_out, true);
				maptri(sisters, standard_landmarks, scaled_sisters, landmarks[i], triangles[k], brothers_in, brothers_out, true);
			}

			save_jpeg(scaled_brothers, "scaled-brothers.jpg");
			save_jpeg(scaled_sisters, "scaled-sisters.jpg");

			matrix<rgb_pixel> altered = img;

			for (size_t x = 0; x < img.nc(); x++) {
				for (size_t y = 0; y < img.nr(); y++) {
					for (size_t a = 0; a < 1; a++) {
						double r = ((double) altered(y, x).red) + scaled_sisters(y, x).red - scaled_brothers(y, x).red;
						double g = ((double) altered(y, x).green) + scaled_sisters(y, x).green - scaled_brothers(y, x).green;
						double b = ((double) altered(y, x).blue) + scaled_sisters(y, x).blue - scaled_brothers(y, x).blue;

						if (r > 255) {
							r = 255;
						}
						if (g > 255) {
							g = 255;
						}
						if (b > 255) {
							b = 255;
						}
						if (r < 0) {
							r = 0;
						}
						if (g < 0) {
							g = 0;
						}
						if (b < 0) {
							b = 0;
						}

						altered(y, x).red = r;
						altered(y, x).green = g;
						altered(y, x).blue = b;
					}
				}
			}

			save_jpeg(altered, "altered.jpg");

			if (reencode) {
				face f2;
				for (size_t j = 0; j < face_descriptors[i].size(); j++) {
					f2.metrics.push_back(face_descriptors[i](j));
				}
				double dist = f.distance(f2);
				if (check_reencode && dist > 0.24) {
					aprintf(ret, "# %0.6f %s\n", dist, fname.c_str());
					continue;
				}
				aprintf(ret, "%0.6f,", dist);
			}

			aprintf(ret, "%zu ", i);

			rectangle rect = landmarks[i].get_rect();

			long width = rect.right() - rect.left();
			long height = rect.bottom() - rect.top();
			aprintf(ret, "%ldx%ld+%ld+%ld", (long) (width / scale), (long) (height / scale), (long) (rect.left() / scale), (long) (rect.top() / scale));

			for (size_t j = 0; j < landmarks[i].num_parts(); j++) {
				point p = landmarks[i].part(j);
				aprintf(ret, " %ld,%ld", (long) (p(0) / scale), (long) (p(1) / scale));
			}

			aprintf(ret, " --");

			for (size_t j = 0; j < face_descriptors[i].size(); j++) {
				aprintf(ret, " %f", face_descriptors[i](j));
			}

			aprintf(ret, " %s\n", fname.c_str());
		}
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

void usage(const char *s) {
	fprintf(stderr, "Usage: %s [-j threads]\n", s);
}

int main(int argc, char **argv) {
	size_t jobs = 1;

	int o;
	extern int optind;
	extern char *optarg;

	while ((o = getopt(argc, argv, "j:flrR")) != -1) {
		switch (o) {
		case 'j':
			jobs = atoi(optarg);
			break;

		case 'f':
			flop = true;
			break;

		case 'l':
			landmarks = true;
			break;

		case 'r':
			reencode = true;
			break;

		case 'R':
			reencode = true;
			check_reencode = true;
			break;

		default:
			usage(*argv);
			exit(EXIT_FAILURE);
		}
	}

	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

	dlib::shape_predictor sp;
	if (landmarks) {
		dlib::deserialize("/usr/local/share/shape_predictor_68_face_landmarks.dat") >> sp;
	} else {
		dlib::deserialize("/usr/local/share/shape_predictor_5_face_landmarks.dat") >> sp;
	}

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
			std::string fname = nextline();
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
