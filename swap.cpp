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
	    size_t triangle[3]) {
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
				img_out(y_out, x_out) = img_in(y_in, x_in);
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

	if (imgs.size() == 1) {
		matrix<rgb_pixel> out = imgs[0];

		for (size_t i = 0; i < landmarkses.size(); i++) {
			size_t j = (i + 1) % (landmarkses.size());

			for (size_t k = 0; k < ntriangles; k++) {
				maptri(imgs[0], landmarkses[j], out, landmarkses[i], triangles[k]);
			}
		}

		char buf[600];
		sprintf(buf, "out-%zu.jpg", (size_t) 0);
		save_jpeg(out, buf);
	} else {
		for (size_t i = 0; i < imgs.size(); i++) {
			size_t j = (i + 1) % (imgs.size());

			matrix<rgb_pixel> out = imgs[i];

			for (size_t k = 0; k < ntriangles; k++) {
				maptri(imgs[j], landmarkses[j], out, landmarkses[i], triangles[k]);
			}

			char buf[600];
			sprintf(buf, "out-%zu.jpg", i);
			save_jpeg(out, buf);
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

	while ((o = getopt(argc, argv, "f")) != -1) {
		switch (o) {
		case 'f':
			flop = true;
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
