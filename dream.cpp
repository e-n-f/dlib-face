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

struct pt {
	int x;
	int y;

	bool operator<(const pt &p) const {
		return y < p.y;
	}
};

void guess(face f, const char *fname) {
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

	dlib::shape_predictor sp;
	dlib::deserialize("/usr/local/share/shape_predictor_5_face_landmarks.dat") >> sp;

	anet_type net;
	dlib::deserialize("/usr/local/share/dlib_face_recognition_resnet_model_v1.dat") >> net;

	matrix<rgb_pixel> img;
	load_image(img, fname);

	double scale = 1;
	while (img.size() > 40000 * 3000 * sqrt(2)) {
		// printf("scale down: %ldx%ld\n", img.nc(), img.nr());
		pyramid_down<2> pyr;
		matrix<rgb_pixel> tmp;
		pyr(img, tmp);
		img = tmp;
		scale /= 2;
	}
	while (img.size() < 500 * 375 / sqrt(2)) {
		// printf("scale up: %ldx%ld\n", img.nc(), img.nr());
		pyramid_up(img);
		scale *= 2;
	}

	double previous = 999;
	rectangle rect(0, 0, img.nc(), img.nr());

	size_t seq = 0;
	while (true) {
		matrix<rgb_pixel> proposed = img;

		bool fail = false;
		size_t iterations = 1;
		for (size_t i = 0; i < iterations; i++) {
			std::vector<pt> pts;

			for (size_t j = 0; j < 2; j++) {
				pt p;
				p.x = std::rand() % (rect.right() - rect.left()) + rect.left();
				p.y = std::rand() % (rect.bottom() - rect.top()) + rect.top();
				pts.push_back(p);
			}

			int ymin = std::min(pts[0].y, pts[1].y);
			int ymax = std::max(pts[0].y, pts[1].y);

			int xmin = std::min(pts[0].x, pts[1].x);
			int xmax = std::max(pts[0].x, pts[1].x);

			double opacity = (std::rand() % 256) / 256.0;
			double value = std::rand() % 256;

			if (1) {
				for (int y = ymin; y < ymax; y++) {
					for (int x = xmin; x < xmax; x++) {
						if (x >= 0 && y >= 0 && x < proposed.nc() && y < proposed.nr()) {
							int r = proposed(y, x).red;
							int g = proposed(y, x).green;
							int b = proposed(y, x).blue;

							r = std::floor(opacity * value + r * (1.0 - opacity));
							g = std::floor(opacity * value + g * (1.0 - opacity));
							b = std::floor(opacity * value + b * (1.0 - opacity));

							proposed(y, x).red = r;
							proposed(y, x).green = g;
							proposed(y, x).blue = b;
						}
					}
				}
			} else {
				if (xmax - xmin == 0 || ymax - ymin == 0) {
					fail = true;
					continue;
				}

				int xc = xmin + std::rand() % (xmax - xmin);
				int yc = ymin + std::rand() % (ymax - ymin);

				if (xc - xmin == 0 || yc - ymin == 0 || ymax - yc == 0 || xmax - xc == 0) {
					fail = true;
					continue;
				}

				for (int y = ymin; y < ymax; y++) {
					for (int x = xmin; x < xmax; x++) {
						int xi, yi;

						if (y < yc) {
							yi = (y - ymin) * ((ymax - ymin) / 2.0) / (yc - ymin) + ymin;
						} else {
							yi = (y - yc) * ((ymax - ymin) / 2.0) / (ymax - yc) + yc;
						}

						if (x < xc) {
							xi = (x - xmin) * ((xmax - xmin) / 2.0) / (xc - xmin) + xmin;
						} else {
							xi = (x - xc) * ((xmax - xmin) / 2.0) / (xmax - xc) + xc;
						}

						if (x >= 0 && y >= 0 && x < img.nc() && y < img.nr() &&
						    xi >= 0 && yi >= 0 && xi < img.nc() && yi < img.nr()) {
							proposed(x, y) = img(xi, yi);
						}
					}
				}
			}
		}

		if (fail) {
			continue;
		}

		matrix<rgb_pixel> face_chip;
		std::vector<full_object_detection> landmarks;

		bool did = false;
		for (auto face : detector(proposed)) {
			full_object_detection shape = sp(proposed, face);
			landmarks.push_back(shape);
			extract_image_chip(proposed, get_face_chip_details(shape, 150, 0.25), face_chip);
			did = true;
		}

		std::vector<matrix<rgb_pixel>> faces;

		if (did && face_chip.nc() == 150) {
			faces.push_back(face_chip);
			std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);

			double dist = 0;
			for (size_t i = 0; i < face_descriptors.size(); i++) {
				for (size_t j = 0; j < face_descriptors[i].size(); j++) {
					double d = face_descriptors[i](j) - f.metrics[j];
					dist += d * d;
				}
			}
			dist = sqrt(dist);

			if (face_descriptors.size() > 0 && dist < previous) {
				previous = dist;
				img = proposed;
				rect = landmarks[0].get_rect();
				fprintf(stderr, "%1.8f \r", dist);

				seq++;
				if (seq % 1 == 0 || dist < 0.1) {
					char buf[600];
					sprintf(buf, "dream.out/%08zu.png", seq);
					save_png(proposed, buf);
				}
			}
		}
	}
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

void read_source(FILE *f, const char *base) {
	std::vector<face> todo;

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
		todo.push_back(fc);
	}

	face avg = mean(todo);
	guess(avg, base);
}

int main(int argc, char **argv) {
	mkdir("dream.out", 0777);
	read_source(stdin, argv[1]);
}
