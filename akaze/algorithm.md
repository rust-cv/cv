# AKAZE explained

At a high level, AKAZE is an algorithm that takes in an image, determines where features
are at different scales in the image, then extracts gradient information for matching at that point.

The first thing AKAZE does is create a scale-space pyramid. This pyramid is a series of
images containing the scale estimation of the image. The higher you go in the pyramid, the scale at which the edges are evaluated changes.

AKAZE does this by setting up a series of three octaves. Each octave is a 2x scaling in the image dimensions, so a 512x512 scale space in octave 0 would become a 256x256 scale space in octave 1. Typically there are three octaves in total: 0, 1, and 2.

AKAZE then further divides these octaves up into sublevels. The typical amount of sublevels to use is four: 0, 1, 2, and 3.

In essence, each `octave,sublevel` pair forms a layer of the pyramid. In the typical case, it looks like this:

0. `0,0`
1. `0,1`
2. `0,2`
3. `0,3`
4. `1,0`
5. `1,1`
6. `1,2`
7. `1,3`
8. `2,0`
9. `2,1`
10. `2,2`
11. `2,3`
