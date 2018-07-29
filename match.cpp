#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <errno.h>

struct face {
	size_t seq;
	std::string bbox;
	std::vector<std::string> landmarks;
	std::vector<float> metrics;
	std::string fname;
};

void usage(const char *s) {
	fprintf(stderr, "Usage: %s [-s subject ...] [candidates ...]\n", s);
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

void read_source(std::string s) {
	FILE *f = fopen(s.c_str(), "r");
	if (f == NULL) {
		fprintf(stderr, "%s: %s\n", s.c_str(), strerror(errno));
		exit(EXIT_FAILURE);
	}

	while (true) {
		std::string s = nextline(f);
		if (s.size() == 0) {
			break;
		}
		s.resize(s.size() - 1);

		face fc = toface(s);
	}

	fclose(f);
}

int main(int argc, char **argv) {
	int i;
	extern int optind;
	extern char *optarg;

	std::vector<std::string> sources;

	while ((i = getopt(argc, argv, "s:")) != -1) {
		switch (i) {
		case 's':
			sources.push_back(optarg);
			break;

		default:
			usage(argv[0]);
			exit(EXIT_FAILURE);
		}
	}

	for (size_t i = 0; i < sources.size(); i++) {
		read_source(sources[i]);
	}
}
