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
        double seq;
        std::string bbox;
        std::vector<std::string> landmarks;
        std::vector<float> metrics;
        std::string fname;
};

struct rgb {
	double r;
	double g;
	double b;

	double mean, m2, stddev;
	size_t count;

	rgb() {
		r = g = b = 0;
		mean = m2 = stddev = 0;
		count = 0;
	}
};

struct xyz {
	double x;
	double y;
	double z;
};

struct lab {
	double l;
	double a;
	double b;
};

struct lch {
	double l;
	double c;
	double h;
};

xyz rgbtoxyz(rgb rgb) {
	double r = rgb.r / 255.0;
	double g = rgb.g / 255.0;
	double b = rgb.b / 255.0;

	// assume sRGB
	if (r <= 0.04045) {
		r = r / 12.92;
	} else {
		r = exp(log((r + 0.055) / 1.055) * 2.4);
	}
	if (g <= 0.04045) {
		g = g / 12.92;
	} else {
		g = exp(log((g + 0.055) / 1.055) * 2.4);
	}
	if (b <= 0.04045) {
		b = b / 12.92;
	} else {
		b = exp(log((b + 0.055) / 1.055) * 2.4);
	}

	r *= 100.0;
	g *= 100.0;
	b *= 100.0;

	xyz ret;

	double M[3][3] = {
		{ 0.4124, 0.3576,  0.1805 },
		{ 0.2126, 0.7152,  0.0722 },
		{ 0.0193, 0.1192,  0.9505 }
	};

	ret.x = (r * M[0][0]) + (g * M[0][1]) + (b * M[0][2]);
	ret.y = (r * M[1][0]) + (g * M[1][1]) + (b * M[1][2]);
	ret.y = (r * M[2][0]) + (g * M[2][1]) + (b * M[2][2]);

	return ret;
}

lab xyztolab(xyz xyz) {
	double whitePoint[3] = { 95.0429, 100.0, 108.8900 };

	double x = xyz.x / whitePoint[0];
	double y = xyz.y / whitePoint[1];
	double z = xyz.z / whitePoint[2];

	if (x > 0.008856) {
		x = exp(log(x) / 3);
	} else {
		x = (7.787 * x) + (16.0 / 116.0);
	}
	if (y > 0.008856) {
		y = exp(log(y) / 3);
	} else {
		y = (7.787 * y) + (16.0 / 116.0);
	}
	if (z > 0.008856) {
		z = exp(log(z) / 3);
	} else {
		z = (7.787 * z) + (16.0 / 116.0);
	}

	lab ret;

	ret.l = (116.0 * y) - 16.0;
	ret.a = 500.0 * (x - y);
	ret.b = 200.0 * (y - z);

	return ret;
}

lch labtolch(lab lab) {
	lch ret;

	ret.l = lab.l;
	ret.c = sqrt(lab.a * lab.a + lab.b * lab.b);
	ret.h = atan2(lab.b, lab.a);

	return ret;
}

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


void guess(face f, std::vector<std::vector<rgb>> &accum, double &count) {
	matrix<rgb_pixel> img;
	try {
		load_image(img, f.fname);
	} catch (...) {
		fprintf(stderr, "can't load %s\n", f.fname.c_str());
		exit(EXIT_FAILURE);
	}

	double weight = 1.0 / (f.seq * f.seq);

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
			accum[x][y].r += weight * face_chip(y, x).red;
			accum[x][y].g += weight * face_chip(y, x).green;
			accum[x][y].b += weight * face_chip(y, x).blue;

			rgb rgb;
			rgb.r = face_chip(y, x).red;
			rgb.g = face_chip(y, x).green;
			rgb.b = face_chip(y, x).blue;

			double light = (face_chip(y, x).blue + 2.0 * face_chip(y, x).red + 4.0 * face_chip(y, x).green) / 7.0;

#if 0
			xyz xyz = rgbtoxyz(rgb);
			lab lab = xyztolab(xyz);
			lch lch = labtolch(lab);

			light = lch.l;
#endif

			// Welford, https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
			accum[x][y].count++;
			double delta = light - accum[x][y].mean;
			accum[x][y].mean += delta / accum[x][y].count;
			double delta2 = light - accum[x][y].mean;
			accum[x][y].m2 += delta * delta2;
			accum[x][y].stddev = sqrt(accum[x][y].m2 / accum[x][y].count);
		}
	}

	count += weight;
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
	f.seq = atof(tok.c_str());

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
	double count = 0;
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

	matrix<rgb_alpha_pixel> pic(SIZE, SIZE);

	double low = 999, high = 0;
	for (size_t x = 0; x < pic.nc(); x++) {
		for (size_t y = 0; y < pic.nr(); y++) {
			pic(y, x).red = (pixels[x][y].r / count - 128) * 1.25 + 128;
			pic(y, x).green = (pixels[x][y].g / count - 128) * 1.25 + 128;
			pic(y, x).blue = (pixels[x][y].b / count - 128) * 1.25 + 128;
			pic(y, x).alpha = 0;

			if (pixels[x][y].stddev < low) {
				low = pixels[x][y].stddev;
			}
			if (pixels[x][y].stddev > high) {
				high = pixels[x][y].stddev;
			}
		}
	}

	printf("low %f, high %f\n", low, high);

	for (size_t x = pic.nc() / 10; x < pic.nc() * .9; x++) {
		for (size_t y = 0; y < pic.nr(); y++) {
			pic(y, x).alpha = 255 - 255.0 * ((pixels[x][y].stddev - low) / (high - low));
			pic(y, x).alpha = exp(log(pic(y, x).alpha / 255.0) / 3) * 255.0;

			double v = (pic(y, x).alpha - 128) * 1 + 128;
			if (v < 0) {
				v = 0;
			}
			if (v > 255) {
				v = 255;
			}
			pic(y, x).alpha = v;

#if 0
			pic(y, x).alpha = 255;
			pic(y, x).red = v;
			pic(y, x).green = v;
			pic(y, x).blue = v;
#endif
		}
	}

	save_png(pic, "out.png");
}

int main(int argc, char **argv) {
	read_source(stdin);
}
