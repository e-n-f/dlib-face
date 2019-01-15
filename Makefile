all: dlib-face-encode dlib-face-match shape_predictor_5_face_landmarks.dat shape_predictor_68_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat dlib-face-dream dlib-face-merge dlib-face-swap make-mean-normalized

install: all
	cp dlib-face-encode /usr/local/bin/dlib-face-encode
	cp dlib-face-match /usr/local/bin/dlib-face-match
	cp extract /usr/local/bin/dlib-face-extract
	cp dlib-face-merge /usr/local/bin/dlib-face-merge
	cp shape_predictor_5_face_landmarks.dat /usr/local/share/shape_predictor_5_face_landmarks.dat
	cp shape_predictor_68_face_landmarks.dat /usr/local/share/shape_predictor_68_face_landmarks.dat
	cp dlib_face_recognition_resnet_model_v1.dat /usr/local/share/dlib_face_recognition_resnet_model_v1.dat
	cp celeba/dlib-celeba-men-avg.encoded /usr/local/share/dlib-celeba-men-avg.encoded
	cp celeba/dlib-celeba-women-avg.encoded /usr/local/share/dlib-celeba-women-avg.encoded
	cp utkface/dlib-utkface-men-avg.encoded /usr/local/share/dlib-utkface-men-avg.encoded
	cp utkface/dlib-utkface-women-avg.encoded /usr/local/share/dlib-utkface-women-avg.encoded
	cp siblings/dlib-siblings-sisters-mean-stddev.encoded /usr/local/share/dlib-siblings-sisters-mean-stddev.encoded
	cp siblings/dlib-siblings-brothers-mean-stddev.encoded /usr/local/share/dlib-siblings-brothers-mean-stddev.encoded
	cp glam/dlib-glam-mean-stddev.encoded /usr/local/share/dlib-glam-mean-stddev.encoded
	cp glam/dlib-noglam-mean-stddev.encoded /usr/local/share/dlib-noglam-mean-stddev.encoded
	cp utkface/dlib-utkface-babies.avg.encoded /usr/local/share/dlib-utkface-babies.avg.encoded
	cp utkface/dlib-utkface-adults.avg.encoded /usr/local/share/dlib-utkface-adults.avg.encoded
	cp dlib-face-mean /usr/local/bin/dlib-face-mean
	cp dlib-face-m2f /usr/local/bin/dlib-face-m2f
	cp dlib-face-f2m /usr/local/bin/dlib-face-f2m
	cp dlib-face-no2glam /usr/local/bin/dlib-face-no2glam
	cp dlib-face-glam2no /usr/local/bin/dlib-face-glam2no
	cp dlib-face-exclude-babies /usr/local/bin/dlib-face-exclude-babies
	cp dlib-face-exclude-adults /usr/local/bin/dlib-face-exclude-adults

shape_predictor_5_face_landmarks.dat.bz2:
	curl -L -O http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2

shape_predictor_68_face_landmarks.dat.bz2:
	curl -L -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

dlib_face_recognition_resnet_model_v1.dat.bz2:
	curl -L -O http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

shape_predictor_5_face_landmarks.dat: shape_predictor_5_face_landmarks.dat.bz2
	bzcat $< > $@

shape_predictor_68_face_landmarks.dat: shape_predictor_68_face_landmarks.dat.bz2
	bzcat $< > $@

dlib_face_recognition_resnet_model_v1.dat: dlib_face_recognition_resnet_model_v1.dat.bz2
	bzcat $< > $@

dlib-face-encode: encode.cpp
	c++ -std=c++11 -L/usr/local/lib -g -Wall -O3 -o $@ $< -ldlib -llapack -lblas -lpng -lz -lgif -ljpeg -lpthread

dlib-face-swap: swap.cpp
	c++ -std=c++11 -L/usr/local/lib -g -Wall -O3 -o $@ $< -ldlib -llapack -lblas -lpng -lz -lgif -ljpeg -lpthread

dlib-face-match: match.cpp
	c++ -std=c++11 -g -Wall -O3 -o $@ $<

dlib-face-dream: dream.cpp
	c++ -std=c++11 -L/usr/local/lib -g -Wall -O3 -o $@ $< -ldlib -llapack -lblas -lpng -lz -lgif -ljpeg -lpthread

dlib-face-merge: merge.cpp
	c++ -std=c++11 -L/usr/local/lib -g -Wall -O3 -o $@ $< -ldlib -llapack -lblas -lpng -lz -lgif -ljpeg -lpthread

make-mean-normalized: make-mean-normalized.cpp
	c++ -std=c++11 -L/usr/local/lib -g -Wall -O3 -o $@ $< -ldlib -llapack -lblas -lpng -lz -lgif -ljpeg -lpthread
