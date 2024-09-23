# Top K Emphasis for Stable Diffusion Webui Forge
Top K Emphasis is advanced method of emphasises which allows more strong conditioning or negative conditioning.
## Syntax
`(text which you want emphasize:1.5c1024-0.5k0.25)`

In above case, "text which you want emphasize" in top 1024th of channels of conditioning will be multiplied by 1.5 and top 25% of channels of output which conditioning processed by to_k will be multiplied by -0.5 .

The syntax is like this:

`"(" <text> ":" ("+"|"-") <weight>[<key>[<threshold>]] [("+"|"-") <weight>[<key>[<threshold>]]]+ ")"`

There are 4 types of key:
- "c" means conditioning will be multiplied by weight on creation.
- "k" means result of to_k(conditioning) in cross attention will be multiplied by weight on every sampling step.
- "v" means result of to_v(conditioning) in cross attention will be multiplied by weight on every sampling step.
- "q" means result of matmul(q, k) in cross attention will be multiplied by weight on every sampling step.
- "s" is similar to "q" but applied after softmax.

If you omit key, it will be interpreted as "c".

Note that using "q", "s" requires to enable Extra Mode which makes slower because of disabling optimizations.

Threshold has 3 ways of interpretations:
- Threshold equal to 0 means all of channels will be multiplied.
- Threshold below 1 means top `(threshold * channels)` th of channels will be multiplied.
- Threshold above or equal to 1 means top threshold th of channels will be multiplied.

If you omit threshold, it will be interpreted as 0.
Note that even for same key, number of channels may vary. e.g. clip_l and clip_g has different number of channels. (768 and 2048 respectively)
## Notes
### Tipical number of channels
- "c": 768 for clip_l and 2048 for clip_g
- "k": out_features of to_k (for SDXL: 640 or 1280)
- "v": out_features of to_v (for SDXL: 640 or 1280)
- "q": For SDXL, for IN04, IN05, OUT03, OUT04, OUT05: `ceil(width_in_px/16)*ceil(height_in_px/16)*number_of_heads` where number_of_heads is 10.
If image size is 1920x1080, number of channels will be 81600.
For IN07, IN08, M00, OUT00, OUT01, OUT02: `ceil(width_in_px/32)*ceil(height_in_px/32)*number_of_heads` where number_of_heads is 20.
If image size is 1920x1080, number of channels will  be 40800.
- "s": Same as "q".