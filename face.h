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
