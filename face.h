#include "mean.h"

#ifndef FACE_
#define FACE_

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
			diff += ((metrics[i] - f.metrics[i]) / f.stddevs[i] * .03) *
			        ((metrics[i] - f.metrics[i]) / f.stddevs[i] * .03);
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
        // Should this average too?
        out.landmarks = inputs[0].landmarks;

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

#endif
