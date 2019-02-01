#ifndef MEAN_
#define MEAN_

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

	void unity() {
		count = 1;
		themean = 1;
		m2 = 1;
	}
};

#endif
