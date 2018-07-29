dlib-face
=========

Dlib installation: https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf

Counter to what it says in http://dlib.net/imaging.html#load_image,
it doesn't seem to be necessary to explicitly build dlib with JPEG support:

```
cmake .. -DDLIB_USE_CUDA=0 -DUSE_AVX_INSTRUCTIONS=1 -DDLIB_JPEG_SUPPORT=1
cmake --build .
make install
```
