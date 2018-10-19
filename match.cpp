#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <errno.h>
#include <math.h>
#include <ctype.h>

double threshold = 3.6;
bool longform = false;
bool scale_stddev = false;

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
	std::vector<float> stddevs;
	std::string fname;

	face minus(face const &f) {
		face ret = *this;
		for (size_t i = 0; i < ret.metrics.size() && i < f.metrics.size(); i++) {
			ret.metrics[i] -= f.metrics[i];
		}
		return ret;
	}

	face plus(face const &f) {
		face ret = *this;
		for (size_t i = 0; i < ret.metrics.size() && i < f.metrics.size(); i++) {
			ret.metrics[i] += f.metrics[i];
		}
		return ret;
	}

	double dot(face const &f) {
		double ret = 0;
		for (size_t i = 0; i < metrics.size() && i < f.metrics.size(); i++) {
			ret += metrics[i] * f.metrics[i];
		}
		return ret;
	}

	face times(double n) {
		face ret = *this;
		for (size_t i = 0; i < ret.metrics.size(); i++) {
			ret.metrics[i] *= n;
		}
		return ret;
	}

	double distance(face const &f) {
		double diff = 0;
		for (size_t i = 0; i < metrics.size() && i < f.metrics.size(); i++) {
			diff += (metrics[i] - f.metrics[i]) * (metrics[i] - f.metrics[i]);
		}
		diff = sqrt(diff);
		return diff;
	}

	double normalized_distance(face const &f) {
		double diff = 0;
		for (size_t i = 0; i < metrics.size() && i < f.metrics.size(); i++) {
			// .03 empirically scales it to about the same as unnormalized
			diff += (metrics[i] - f.metrics[i]) * (metrics[i] - f.metrics[i]) / f.stddevs[i] * .03;
		}
		diff = sqrt(diff);
		return diff;
	}

	double magnitude() {
		double diff = 0;
		for (size_t i = 0; i < metrics.size(); i++) {
			diff += metrics[i] * metrics[i];
		}
		diff = sqrt(diff);
		return diff;
	}
};

std::vector<face> subjects;
std::vector<face> origins;
std::vector<face> destinations;
std::vector<face> exclude;

bool goodonly = false;

void usage(const char *s) {
	fprintf(stderr, "Usage: %s [-g] [-s subject ...] [-o origin -d destination] [candidates ...]\n", s);
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
		f.stddevs.push_back(1);
	}

	f.fname = s;
	return f;
}

face mean(std::vector<face> inputs) {
	if (inputs.size() == 0) {
		fprintf(stderr, "Trying to average empty inputs\n");
		exit(EXIT_FAILURE);
	}

	std::vector<mean_stddev> accum;
	accum.resize(inputs[0].metrics.size());

	for (size_t i = 0; i < inputs.size(); i++) {
		for (size_t j = 0; j < inputs[i].metrics.size(); j++) {
			if (j >= accum.size()) {
				fprintf(stderr, "%s: too many metrics\n", inputs[i].fname.c_str());
				exit(EXIT_FAILURE);
			}
			accum[j].add(inputs[i].metrics[j]);
		}
	}

	face out;
	for (size_t i = 0; i < accum.size(); i++) {
		out.metrics.push_back(accum[i].mean());
		out.stddevs.push_back(accum[i].stddev());
	}

	return out;
}

void read_source(std::string s, std::vector<face> &out) {
	FILE *f = fopen(s.c_str(), "r");
	if (f == NULL) {
		fprintf(stderr, "%s: %s\n", s.c_str(), strerror(errno));
		exit(EXIT_FAILURE);
	}

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
	avg.fname = s;

	out.push_back(avg);

	fclose(f);
}

size_t count = 0;
double themean = 0;
double m2 = 0;

void compare(face a, face b, std::string orig) {
	if (a.metrics.size() != b.metrics.size()) {
		fprintf(stderr, "%s: %s: mismatched metrics\n", a.fname.c_str(), b.fname.c_str());
		return;
	}

	double diff;

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
					if (longform) {
						printf("%01.6f,%01.6f,%01.6f %s\n", along - canonalong, dist, along, orig.c_str());
					} else {
						printf("%01.6f,%01.6f,%01.6f\t%s\t%s\t%s\t%s\n", along - canonalong, dist, along, a.fname.c_str(), a.bbox.c_str(), b.fname.c_str(), b.bbox.c_str());
					}

					if (goodonly) {
						fflush(stdout);
					}
				}
			}
		}
	}
}

void read_candidates(FILE *fp) {
	while (true) {
		std::string s = nextline(fp);
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

	while ((i = getopt(argc, argv, "s:go:d:t:x:ln")) != -1) {
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

		case 'n':
			scale_stddev = true;
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
		for (; optind < argc; optind++) {
			FILE *f = fopen(argv[optind], "r");
			if (f == NULL) {
				fprintf(stderr, "%s: %s: %s\n", argv[0], argv[optind], strerror(errno));
				exit(EXIT_FAILURE);
			}

			read_candidates(f);
			fclose(f);
		}
	}

	fprintf(stderr, "%zu: %.6f %.6f\n", count, themean, sqrt(m2 / count));
}
