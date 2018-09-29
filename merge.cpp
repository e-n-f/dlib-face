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


void guess(face f) {
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
	extract_image_chip(img, get_face_chip_details(shape, 300, 0.25), face_chip);
	save_jpeg(face_chip, "out.jpg");
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

	printf("we have %s\n", s.c_str());
	f.fname = s;
	return f;
}


face mean(std::vector<face> inputs) {
	face out;
	size_t count = 0;

	for (size_t i = 0; i < inputs.size(); i++) {
		if (i == 0) {
			out.metrics = inputs[i].metrics;
			count = 1;
		} else {
			for (size_t j = 0; j < inputs[i].metrics.size(); j++) {
				if (j >= out.metrics.size()) {
					fprintf(stderr, "%s: too many metrics\n", inputs[i].fname.c_str());
					exit(EXIT_FAILURE);
				}
				out.metrics[j] += inputs[i].metrics[j];
			}
			count++;
		}
	}

	for (size_t i = 0; i < out.metrics.size(); i++) {
		out.metrics[i] /= count;
	}

	return out;
}

void read_source(FILE *f) {
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
		guess(fc);
	}
}

int main(int argc, char **argv) {
	read_source(stdin);
}
