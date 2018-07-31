all: dlib-face-encode dlib-face-match shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat dlib-face-dream

install: all
	cp dlib-face-encode /usr/local/bin/dlib-face-encode
	cp dlib-face-match /usr/local/bin/dlib-face-match
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
	c++ -std=c++11 -L/usr/local/lib -g -Wall -O3 -o $@ $< -ldlib -llapack -lblas -lpng -lz -lgif -ljpeg -lpthread

dlib-face-match: match.cpp
	c++ -std=c++11 -g -Wall -O3 -o $@ $<

dlib-face-dream: dream.cpp
	c++ -std=c++11 -L/usr/local/lib -g -Wall -O3 -o $@ $< -ldlib -llapack -lblas -lpng -lz -lgif -ljpeg -lpthread
