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
#include <atomic>

using namespace dlib;

bool flop = false;
bool landmarks = false;
bool reencode = false;
bool check_reencode = false;
bool cropped = false;
char *cwd;
long pixels = 1024;
bool do_jitter = false;
const char *extract_file = NULL;

std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
)
{
    // All this function does is make 25 copies of img, all slightly jittered by being
    // zoomed, rotated, and translated a little bit differently. They are also randomly
    // mirrored left to right.
    thread_local dlib::rand rnd;

    std::vector<matrix<rgb_pixel>> crops; 
    for (int i = 0; i < 25; ++i)
        crops.push_back(jitter_image(img,rnd));

    return crops;
}

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
		if (tok == "--" || tok == "") {
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

std::atomic<size_t> num(0);

void *run1(void *v) {
	arg *a = (arg *) v;

	std::vector<std::string> *fnames = a->fnames;
	dlib::frontal_face_detector *detector = a->detector;
	dlib::shape_predictor *sp = a->sp;
	anet_type *net = a->net;

	std::string ret;

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
			fname = std::string(cwd) + "/" + fname;
		}

		try {
			load_image(img, fname);
		} catch (...) {
			fprintf(stderr, "%s: failed image loading\n", fname.c_str());
			continue;
		}

		double scale = 1;
		long pixels2 = pixels * 3/4;

		while (img.size() > pixels * pixels2 * sqrt(2)) {
			// printf("scale down: %ldx%ld\n", img.nc(), img.nr());
			pyramid_down<2> pyr;
			matrix<rgb_pixel> tmp;
			pyr(img, tmp);
			img = tmp;
			scale /= 2;
		}

		// 512 finds 27% as many people in 37.5% of the time
		// 1024 runs at reasonable speed
		// 2048 finds 2.64x as many people as 1024, reasonable quality, in 3.7x the time
		// 4096 finds 4.29x as many people as 1024, many low quality, in 13.75x the time

		while (img.size() < pixels * pixels2 / sqrt(2)) {
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

				if (extract_file) {
					matrix<rgb_pixel> f;
					extract_image_chip(img, get_face_chip_details(shape, 600, 0.75), f);
					std::string out = std::string(extract_file) + "-" + std::to_string(num) + ".png";
					num++;
					save_png(f, out.c_str());
				}
			} else {
				fprintf(stderr, "Can't parse bounding box %s for %s\n", f.bbox.c_str(), fname.c_str());
			}
		} else {
			if (cropped) {
				rectangle face(img.nc() / 4, img.nr() / 4, img.nc() * 3 / 4, img.nr() * 3 / 4);

				full_object_detection shape = (*sp)(img, face);
				landmarks.push_back(shape);

				matrix<rgb_pixel> face_chip;
				extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);

				faces.push_back(std::move(face_chip));

				if (extract_file) {
					matrix<rgb_pixel> f;
					extract_image_chip(img, get_face_chip_details(shape, 600, 0.75), f);
					std::string out = std::string(extract_file) + "-" + std::to_string(num) + ".png";
					num++;
					save_png(f, out.c_str());
				}
			} else {
				for (auto face : (*detector)(img)) {
					full_object_detection shape = (*sp)(img, face);
					landmarks.push_back(shape);

					matrix<rgb_pixel> face_chip;
					extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);

					faces.push_back(std::move(face_chip));

					if (extract_file) {
						matrix<rgb_pixel> f;
						extract_image_chip(img, get_face_chip_details(shape, 600, 0.75), f);
						std::string out = std::string(extract_file) + "-" + std::to_string(num) + ".png";
						num++;
						save_png(f, out.c_str());
					}
				}
			}
		}

		std::vector<matrix<float, 0, 1>> face_descriptors;
		for (auto &f : faces) {
			if (do_jitter) {
				face_descriptors.push_back(mean(mat((*net)(jitter_image(f)))));
			} else {
				face_descriptors.push_back((*net)(f));
			}
		}

		if (faces.size() != landmarks.size()) {
			aprintf(ret, "%zu faces but %zu landmarks\n", faces.size(), landmarks.size());
			continue;
		}

		if (faces.size() == 0) {
			aprintf(ret, "# %s\n", fname.c_str());
		}

		for (size_t i = 0; i < face_descriptors.size(); i++) {
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

			if (flop) {
				aprintf(ret, "%ldx%ld+%ld+%ld", (long) (width / scale), (long) (height / scale), (long) ((img.nc() - 1 - rect.right()) / scale), (long) (rect.top() / scale));
			} else {
				aprintf(ret, "%ldx%ld+%ld+%ld", (long) (width / scale), (long) (height / scale), (long) (rect.left() / scale), (long) (rect.top() / scale));
			}

			for (size_t j = 0; j < landmarks[i].num_parts(); j++) {
				point p = landmarks[i].part(j);

				if (flop) {
					aprintf(ret, " %ld,%ld", (long) ((img.nc() - 1 - p(0)) / scale), (long) (p(1) / scale));
				} else {
					aprintf(ret, " %ld,%ld", (long) (p(0) / scale), (long) (p(1) / scale));
				}
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

	cwd = getcwd(NULL, 0);
	if (cwd == NULL) {
		perror("getcwd");
		exit(EXIT_FAILURE);
	}

	while ((o = getopt(argc, argv, "j:flrRcp:Je:")) != -1) {
		switch (o) {
		case 'j':
			jobs = atoi(optarg);
			break;

		case 'J':
			do_jitter = true;
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

		case 'c':
			cropped = true;
			break;

		case 'p':
			pixels = atoi(optarg);
			break;

		case 'e':
			extract_file = optarg;
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
