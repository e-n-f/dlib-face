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
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <getopt.h>
#include <pthread.h>
#include <stdarg.h>
#include <unistd.h>
#include "face.h"
#include "mean.h"

using namespace dlib;

size_t triangles[][3] = {
	#include "triangulation.h"
};
size_t ntriangles = (sizeof(triangles) / (3 * sizeof(size_t)));

double landmark_pixels[][2] = {
	#include "landmarks-68.h"
};

double brother_pixels[][2] = {
	#include "brothers-68.h"
};

double sister_pixels[][2] = {
	#include "sisters-68.h"
};

bool flop = false;
bool landmarks = true;
bool reencode = false;
bool check_reencode = false;

bool wantmale = false;
bool wantfemale = false;
bool wantneutral = false;

std::vector<face> origins;
std::vector<face> destinations;

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

double along_spectrum(face &a, face &origin, face &destination) {
	// following https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Vector_formulation
	face A = origin; // the reference face
	face P = a; // the face we are interested in
	face P2 = origin; // the canonical origin

	face N = destination.minus(origin); // vector along the spectrum
	double magnitude_to_dest = N.magnitude();
	N = N.times(1 / magnitude_to_dest); // make into unit vector

	face AminusP = A.minus(P);
	double AminusPdotN = AminusP.dot(N);
	face AminusPdotNtimesN = N.times(AminusPdotN);

	face closest = A.minus(AminusPdotNtimesN);
	double dist = AminusP.minus(AminusPdotNtimesN).magnitude();
	double along = 5 - AminusPdotN / magnitude_to_dest;

	double canonalong;
	{
		face AminusP2 = A.minus(P2);
		double AminusP2dotN = AminusP2.dot(N);
		face AminusP2dotNtimesN = N.times(AminusP2dotN);

		face closest = A.minus(AminusP2dotNtimesN);
		canonalong = - AminusP2dotN / magnitude_to_dest;
	}

	return along - canonalong;
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
		load_image(brothers, "/usr/local/share/dlib-siblings-brothers.png");
	} catch (...) {
		fprintf(stderr, "brothers: failed image loading\n");
		exit(EXIT_FAILURE);
	}

	try {
		load_image(sisters, "/usr/local/share/dlib-siblings-sisters.png");
	} catch (...) {
		fprintf(stderr, "sisters: failed image loading\n");
		exit(EXIT_FAILURE);
	}

	if (true) {
		// Normalize lightnesses along the forehead transition

		double r1b = brothers(34, 305).red;
		double g1b = brothers(34, 305).green;
		double b1b = brothers(34, 305).blue;

		double r2b = brothers(367, 304).red;
		double g2b = brothers(367, 304).green;
		double b2b = brothers(367, 304).blue;

		double r1s = sisters(34, 305).red;
		double g1s = sisters(34, 305).green;
		double b1s = sisters(34, 305).blue;

		double r2s = sisters(367, 304).red;
		double g2s = sisters(367, 304).green;
		double b2s = sisters(367, 304).blue;

		for (size_t y = 0; y < sisters.nr(); y++) {
			for (size_t x = 0; x < sisters.nc(); x++) {
				sisters(y, x).red = (sisters(y, x).red -     r1s) + r1b;
				sisters(y, x).green = (sisters(y, x).green - g1s) + g1b;
				sisters(y, x).blue = (sisters(y, x).blue -   b1s) + b1b;
			}
		}
	}

	rectangle brother_rect(0, 0, 612, 612);
	std::vector<point> brother_landmarks_v;
	for (size_t i = 0; i < 68; i++) {
		point p(landmark_pixels[i][0], brother_pixels[i][1]);
		brother_landmarks_v.push_back(p);
	}
	full_object_detection brother_landmarks(brother_rect, brother_landmarks_v);

	rectangle sister_rect(0, 0, 612, 612);
	std::vector<point> sister_landmarks_v;
	for (size_t i = 0; i < 68; i++) {
		point p(landmark_pixels[i][0], sister_pixels[i][1]);
		sister_landmarks_v.push_back(p);
	}
	full_object_detection sister_landmarks(sister_rect, sister_landmarks_v);

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
			double prev_gender;
			{
				face f2;
				for (size_t j = 0; j < face_descriptors[i].size(); j++) {
					f2.metrics.push_back(face_descriptors[i](j));
				}

				prev_gender = along_spectrum(f2, origins[0], destinations[0]);

			}

			fprintf(stderr, "initial: %0.2f\n", prev_gender);
			bool male;
			if (prev_gender > 5.5) {
				male = true;
			} else {
				male = false;
			}

			if (wantmale) {
				male = true;
			}
			if (wantfemale) {
				male = false;
			}

			double mult = 0.25;

			while (true) {
				matrix<rgb_pixel> altered = img;

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
				for (size_t k = 0; k < 3; k++) {
					brothers_in[k].unity();
					brothers_out[k].unity();
					sisters_in[k].unity();
					sisters_out[k].unity();
				}

				for (size_t k = 0; k < ntriangles; k++) {
					maptri(brothers, standard_landmarks, scaled_brothers, landmarks[i], triangles[k], brothers_in, brothers_out, true);
					maptri(sisters, standard_landmarks, scaled_sisters, landmarks[i], triangles[k], brothers_in, brothers_out, true);
				}

				// save_jpeg(scaled_brothers, "scaled-brothers.jpg");
				// save_jpeg(scaled_sisters, "scaled-sisters.jpg");

				for (size_t x = 0; x < img.nc(); x++) {
					for (size_t y = 0; y < img.nr(); y++) {
						for (size_t a = 0; a < 1; a++) {
							double r, g, b;

							if (male) {
								r = ((double) img(y, x).red) + mult * (scaled_brothers(y, x).red - scaled_sisters(y, x).red);
								g = ((double) img(y, x).green) + mult * (scaled_brothers(y, x).green - scaled_sisters(y, x).green);
								b = ((double) img(y, x).blue) + mult * (scaled_brothers(y, x).blue - scaled_sisters(y, x).blue);
							} else {
								r = ((double) img(y, x).red) + mult * (scaled_sisters(y, x).red - scaled_brothers(y, x).red);
								g = ((double) img(y, x).green) + mult * (scaled_sisters(y, x).green - scaled_brothers(y, x).green);
								b = ((double) img(y, x).blue) + mult * (scaled_sisters(y, x).blue - scaled_brothers(y, x).blue);
							}

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

				// Reencode face for revised image and landmarks

				matrix<rgb_pixel> face_chip;
				extract_image_chip(altered, get_face_chip_details(landmarks[i], 150, 0.25), face_chip);
				std::vector<matrix<rgb_pixel>> fcs;
				fcs.push_back(face_chip);

				std::vector<matrix<float, 0, 1>> fd2 = (*net)(fcs);
				face_descriptors[i] = fd2[0];

				const char *out = fname.c_str();
				for (const char *cp = out; *cp != '\0'; cp++) {
					if (*cp == '/') {
						out = cp + 1;
					}
				}

				// printf("%s\n", out2.c_str());

				face f2;
				for (size_t j = 0; j < face_descriptors[i].size(); j++) {
					f2.metrics.push_back(face_descriptors[i](j));
				}

				if (reencode) {
					double dist = f.distance(f2);
					if (check_reencode && dist > 0.24) {
						aprintf(ret, "# %0.6f %s\n", dist, fname.c_str());
						continue;
					}
					aprintf(ret, "%0.6f,", dist);
				}

				double gender_out = along_spectrum(f2, origins[0], destinations[0]);
				fprintf(stderr, "gender: %f with %f\n", gender_out, mult);

				if ((wantneutral &&
					((!male && gender_out < 5.5) ||
					 ( male && gender_out > 5.5))) ||
				    (!wantneutral &&
					((!male && gender_out < 5.9) ||
					 ( male && gender_out > 5.1)))) {
					if ((!male && gender_out < prev_gender) ||
					    (male && gender_out > prev_gender)) {
						fprintf(stderr, "regressing, now %f at %f\n", gender_out, mult);
					} else {
						prev_gender = gender_out;
						mult = mult + 0.25;
						continue;
					}
				}

				std::string out2 = std::string(out) + "-gender.jpg";
				save_jpeg(altered, out2.c_str());
				img = altered;

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

				aprintf(ret, " %s\n", out2.c_str());

				break;
			}
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

	while ((o = getopt(argc, argv, "j:lrRmfn")) != -1) {
		switch (o) {
		case 'j':
			jobs = atoi(optarg);
			break;

		case 'm':
			wantmale = true;
			break;

		case 'f':
			wantfemale = true;
			break;

		case 'n':
			wantneutral = true;
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

	std::vector<std::string> origin_files, destination_files;

	origin_files.push_back("/usr/local/share/dlib-siblings-brothers-mean-stddev.encoded");
	destination_files.push_back("/usr/local/share/dlib-siblings-sisters-mean-stddev.encoded");

        for (size_t i = 0; i < origin_files.size(); i++) {
                read_source(origin_files[i], origins, true);
        }

        for (size_t i = 0; i < destination_files.size(); i++) {
                read_source(destination_files[i], destinations, true);
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
			std::string fname = nextline(stdin);
			seq++;

			if (fname.size() == 0) {
				break;
			}

			fname.resize(fname.size() - 1);
			jobq[seq % jobs].fnames->push_back(fname);

			if (jobq[seq % jobs].fnames->size() > 0) {
				runq(jobq);
			}
		}
	} else {
		for (; optind < argc; optind++) {
			jobq[optind % jobs].fnames->push_back(argv[optind]);

			if (jobq[optind % jobs].fnames->size() > 0) {
				runq(jobq);
			}
		}
	}

	runq(jobq);
}
