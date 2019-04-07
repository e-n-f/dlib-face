#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <errno.h>
#include <math.h>
#include <ctype.h>
#include <sys/stat.h>
#include "face.h"
#include "mean.h"

double threshold = 3.6;
bool longform = false;
bool scale_stddev = false;
bool adjust = false;
bool landmark_similarity = false;
bool no_subject = false;

size_t total_bytes = 0;
size_t along = 0;
size_t seq = 0;

std::vector<face> subjects;
std::vector<face> origins;
std::vector<face> destinations;
std::vector<face> exclude;

bool goodonly = false;

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

void usage(const char *s) {
	fprintf(stderr, "Usage: %s [-g] [-s subject ...] [-o origin -d destination] [candidates ...]\n", s);
}

size_t count = 0;
double themean = 0;
double m2 = 0;
size_t accepted = 0;

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
		double d = sqrt(xd * xd + yd * yd);
		badness += d;
	}

	return badness;
}

void compare(face a, face b, std::string orig) {
	if (a.metrics.size() != b.metrics.size()) {
		fprintf(stderr, "%s: %s: mismatched metrics\n", a.fname.c_str(), b.fname.c_str());
		return;
	}

	seq++;
	if (isatty(2) && seq % 5000 == 0 && total_bytes != 0) {
		fprintf(stderr, "%3.1f%%: %zu\r", 100.0 * along / total_bytes, seq);
	}

	double diff;

	if (landmark_similarity) {
		printf("%.6f,", calc_landmark_similarity(a, b));
	}

	if (scale_stddev) {
		diff = a.normalized_distance(b);
	} else {
		diff = a.distance(b);
	}

	if (1) {
		if (origins.size() == 0) {
			count++;
			double delta = diff - themean;
			themean = themean + delta / count;
			double delta2 = diff - themean;
			m2 = m2 + delta * delta2;
			double stddev = sqrt(m2 / count);

			bool excluded = false;
			for (size_t i = 0; i < exclude.size(); i++) {
				double diff2 = a.distance(exclude[i]);

				if (diff2 < diff) {
					excluded = true;
					break;
				}
			}

			if (excluded) {
				return;
			}

			if (!goodonly || diff < themean - threshold * stddev) {
				if (longform) {
					printf("%01.6f %s\n", diff, orig.c_str());
				} else {
					printf("%01.6f\t%s\t%s\t%s\t%s\n", diff, a.fname.c_str(), a.bbox.c_str(), b.fname.c_str(), b.bbox.c_str());
				}

				accepted++;

				if (goodonly) {
					fflush(stdout);
				}
			}
		} else {
			// following https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Vector_formulation
			face A = b; // the reference face
			face P = a; // the face we are interested in
			face P2 = origins[0]; // the canonical origin

			for (size_t i = 0; i < origins.size(); i++) {
				face N = destinations[i].minus(origins[i]); // vector along the spectrum
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

				count++;
				double delta = dist - themean;
				themean = themean + delta / count;
				double delta2 = dist - themean;
				m2 = m2 + delta * delta2;
				double stddev = sqrt(m2 / count);

				if (!goodonly || dist < themean - threshold * stddev) {
					if (!no_subject) {
						printf("%01.6f,", dist);
					}

					if (longform) {
						printf("%01.6f %s\n", along - canonalong, orig.c_str());
					} else {
						printf("%01.6f\t%s\t%s\t%s\t%s\n", along - canonalong, a.fname.c_str(), a.bbox.c_str(), b.fname.c_str(), b.bbox.c_str());
					}

					accepted++;

					if (goodonly) {
						fflush(stdout);
					}
				}
			}
		}

		if (goodonly && adjust && (count % 1000 == 0)) {
			if ((double) accepted / count < .0005 / 1.05) {
				threshold /= 1.01;
				fprintf(stderr, "%zu/%zu: threshold now %0.5f\r", accepted, count, threshold);
			}
			if ((double) accepted / count > .0005 * 1.05) {
				threshold *= 1.01;
				fprintf(stderr, "%zu/%zu: threshold now %0.5f\r", accepted, count, threshold);
			}
		}
	}
}

void read_candidates(FILE *fp) {
	while (true) {
		std::string s = nextline(fp);
		along += s.size();
		if (s.size() == 0) {
			break;
		}
		if (!isdigit(s[0])) {
			continue;
		}
		s.resize(s.size() - 1);

		face fc = toface(s);

		const char *orig = s.c_str();
		while (*orig && !isspace(*orig)) {
			orig++;
		}
		while (*orig && isspace(*orig)) {
			orig++;
		}

		for (size_t i = 0; i < subjects.size(); i++) {
			compare(fc, subjects[i], orig);
		}
	}
}

int main(int argc, char **argv) {
	int i;
	extern int optind;
	extern char *optarg;

	std::vector<std::string> sources;
	std::vector<std::string> origin_files;
	std::vector<std::string> destination_files;
	std::vector<std::string> exclude_files;

	while ((i = getopt(argc, argv, "s:go:d:t:x:lnGaL")) != -1) {
		switch (i) {
		case 's':
			sources.push_back(optarg);
			break;

		case 'o':
			origin_files.push_back(optarg);
			break;

		case 'd':
			destination_files.push_back(optarg);
			break;

		case 'x':
			exclude_files.push_back(optarg);
			break;

		case 'g':
			goodonly = true;
			break;

		case 't':
			threshold = atof(optarg);
			break;

		case 'l':
			longform = true;
			break;

		case 'L':
			landmark_similarity = true;
			break;

		case 'n':
			scale_stddev = true;
			break;

		case 'G':
			origin_files.push_back("/usr/local/share/dlib-siblings-brothers-mean-stddev.encoded");
			destination_files.push_back("/usr/local/share/dlib-siblings-sisters-mean-stddev.encoded");
			break;

		case 'a':
			adjust = true;
			break;

		default:
			usage(argv[0]);
			exit(EXIT_FAILURE);
		}
	}

	for (size_t i = 0; i < sources.size(); i++) {
		read_source(sources[i], subjects);
	}

	for (size_t i = 0; i < origin_files.size(); i++) {
		read_source(origin_files[i], origins);
	}

	for (size_t i = 0; i < destination_files.size(); i++) {
		read_source(destination_files[i], destinations);
	}

	for (size_t i = 0; i < exclude_files.size(); i++) {
		read_source(exclude_files[i], exclude);
	}

	if (destinations.size() != 0 && subjects.size() == 0) {
		subjects.push_back(destinations[0]);
		no_subject = true;
	}

	if (subjects.size() == 0) {
		fprintf(stderr, "%s: No subjects specified\n", *argv);
		exit(EXIT_FAILURE);
	}

	if (destinations.size() != origins.size()) {
		fprintf(stderr, "%s: -o and -d must be used together\n", *argv);
		exit(EXIT_FAILURE);
	}

	if (optind == argc) {
		if (isatty(0)) {
			fprintf(stderr, "Warning: standard input is a terminal\n");
		}

		read_candidates(stdin);
	} else {
		for (size_t i = optind; i < argc; i++) {
			struct stat st;
			if (stat(argv[i], &st) == 0) {
				total_bytes += st.st_size;
			}
		}

		for (size_t i = optind; i < argc; i++) {
			FILE *f = fopen(argv[i], "r");
			if (f == NULL) {
				fprintf(stderr, "%s: %s: %s\n", argv[0], argv[i], strerror(errno));
				exit(EXIT_FAILURE);
			}

			read_candidates(f);
			fclose(f);
		}
	}

	fprintf(stderr, "%zu: %.6f %.6f\n", count, themean, sqrt(m2 / count));
}
