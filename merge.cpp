// Following:
//
// http://dlib.net/dnn_face_recognition_ex.cpp.html
// https://github.com/ageitgey/face_recognition/blob/master/face_recognition/api.py

// for vasprintf() on Linux
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#define SIZE 500

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
#include <vector>
#include <algorithm>

using namespace dlib;

struct face {
        size_t seq;
        std::string bbox;
        std::vector<std::string> landmarks;
        std::vector<float> metrics;
        std::string fname;
};

struct rgb {
	long r;
	long g;
	long b;

	rgb() {
		r = g = b = 0;
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

#if 0

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

#endif


void guess(face f, std::vector<std::vector<rgb>> &accum, size_t &count) {
	matrix<rgb_pixel> img;
	try {
		load_image(img, f.fname);
	} catch (...) {
		fprintf(stderr, "can't load %s\n", f.fname.c_str());
		exit(EXIT_FAILURE);
	}

	std::vector<point> points;
	for (size_t i = 0; i < f.landmarks.size(); i++) {
		point p;
		long x, y;

		if (sscanf(f.landmarks[i].c_str(), "%ld,%ld", &x, &y) != 2) {
			fprintf(stderr, "Can't parse landmark %s\n", f.landmarks[i].c_str());
			exit(EXIT_FAILURE);
		}

		p(0) = x;
		p(1) = y;
		points.push_back(p);
	}

	long left, top, width, height;
	if (sscanf(f.bbox.c_str(), "%ldx%ld+%ld+%ld", &width, &height, &left, &top) != 4) {
		fprintf(stderr, "Can't parse bbox %s\n", f.bbox.c_str());
		exit(EXIT_FAILURE);
	}
	rectangle r(left, top, left + width, top + height);

	full_object_detection shape(r, points);

	matrix<rgb_pixel> face_chip;
	extract_image_chip(img, get_face_chip_details(shape, SIZE, 0.90), face_chip);

	for (size_t x = 0; x < face_chip.nc(); x++) {
		for (size_t y = 0; y < face_chip.nr(); y++) {
			accum[x][y].r += face_chip(y, x).red;
			accum[x][y].g += face_chip(y, x).green;
			accum[x][y].b += face_chip(y, x).blue;
		}
	}

	count++;
}

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

	for (size_t i = 0; i < 5; i++) {
		tok = gettok(s);
		f.landmarks.push_back(tok);
	}

	tok = gettok(s); // --

	for (size_t i = 0; i < 128; i++) {
		tok = gettok(s);
		f.metrics.push_back(atof(tok.c_str()));
	}

	f.fname = s;
	return f;
}

void read_source(FILE *f) {
	size_t count = 0;
	std::vector<std::vector<rgb>> pixels;
	pixels.resize(SIZE);
	for (size_t i = 0; i < SIZE; i++) {
		pixels[i].resize(SIZE);
	}

	while (true) {
		std::string s = nextline(f);
		if (s.size() == 0) {
			break;
		}
		if (!isdigit(s[0])) {
			continue;
		}
		s.resize(s.size() - 1);

		face fc = toface(s);
		guess(fc, pixels, count);
	}

	matrix<rgb_pixel> pic(SIZE, SIZE);

	for (size_t x = 0; x < pic.nc(); x++) {
		for (size_t y = 0; y < pic.nr(); y++) {
			pic(y, x).red = pixels[x][y].r / count;
			pic(y, x).green = pixels[x][y].g / count;
			pic(y, x).blue = pixels[x][y].b / count;
		}
	}

	save_jpeg(pic, "out.jpg");
}

int main(int argc, char **argv) {
	read_source(stdin);
}
