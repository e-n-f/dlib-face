all: dlib-face-encode dlib-face-match shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat

install: all
	cp dlib-face-encode /usr/local/bin/dlib-face-encode
	cp shape_predictor_5_face_landmarks.dat /usr/local/share/shape_predictor_5_face_landmarks.dat
	cp dlib_face_recognition_resnet_model_v1.dat /usr/local/share/dlib_face_recognition_resnet_model_v1.dat

shape_predictor_5_face_landmarks.dat.bz2:
	curl -L -O http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2

dlib_face_recognition_resnet_model_v1.dat.bz2:
	curl -L -O http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

shape_predictor_5_face_landmarks.dat: shape_predictor_5_face_landmarks.dat.bz2
	bzcat $< > $@

dlib_face_recognition_resnet_model_v1.dat: dlib_face_recognition_resnet_model_v1.dat.bz2
	bzcat $< > $@

dlib-face-encode: encode.cpp
	c++ -std=c++14 -g -Wall -O3 -o $@ $< -ldlib -llapack -lcblas -lpng -lz -lgif -ljpeg

dlib-face-match: match.cpp
	c++ -std=c++14 -g -Wall -O3 -o $@ $<
