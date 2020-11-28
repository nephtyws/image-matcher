# image-matcher
Fast similar/identical image matcher based on perceptual hash and HSV color hash

## Observation
I have applied these methods to match identical or very similar images between two sets of clothes/fashion images. The first method, perceptual hash, is very robust on the identical images but it can't match a very similar image. But it can be used to categorize or cluster the similar images in a one set. On the other hand, the color hash, is working more effectively to match a very similar image but it generates too many hash collisions (so the method can be confused if there are many similar colored images) so it might not be inappropriate to apply on the large image set. and it can also be used to categorize or cluster the similar images as the first one does. In summary, I can acquire moderated results when I combine two methods as shown to the code.

## References
- https://tech.okcupid.com/evaluating-perceptual-image-hashes-okcupid/
- https://github.com/JohannesBuchner/imagehash
