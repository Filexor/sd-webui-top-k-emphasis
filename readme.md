# Top K Emphasis for Stable Diffusion Webui Forge
Top K Emphasis is advanced method of emphasises which allows more strong conditioning or negative conditioning.
## Syntax
`(text which you want emphasize:1.5c1024-0.5k0.25)`

In above case, "text which you want emphasize" in top 1024th of channels of conditioning will be multiplied by 1.5 and top 25% of channels of output which conditioning processed by to_k will be multiplied by -0.5 .

The syntax is like this:

`"(" <text> ":" ("+"|"-") <weight>[<key>[<threshold>[<option>[<value>]]]] [("+"|"-") <weight>[<key>[<threshold>[<option>[<value>]]]]]+ ")"`

There are 12 types of key:
- "b": Emphasis will be applied after enumeration of embeddings.
- "bl": Simillar to "b", but only for "clip_l". Both SD1.5 and SDXL uses "clip_l".
- "bg": Simillar to "b", but only for "clip_g". SDXL also uses "clip_g".
- "c": Emphasis will be applied after embeddings being converted to conditioning.
- "l": Simillar to "c", but only for "clip_l". Both SD1.5 and SDXL uses "clip_l".
- "g": Simillar to "c", but only for "clip_g". SDXL also uses "clip_g".
- "pk": Emphasis will be applied before conditioning being feeded to each to_k in cross attention.
- "k": Emphasis will be applied after conditioning being feeded to each to_k in cross attention.
- "pv": Emphasis will be applied before conditioning being feeded to each to_v in cross attention.
- "v": Emphasis will be applied after conditioning being feeded to each to_v in cross attention.
- "q": Emphasis will be applied after each `torch.einsum('b i d, b j d -> b i j', q, k)` in cross attention.
- "s": Similar to "q" but applied after softmax.

If you omit key, it will be interpreted as "c".

Note that using "q", "qh", "ql", "s", "sh", "sl" requires to enable Extra Mode which makes slower because of disabling optimizations.

Threshold has 3 ways of interpretations:
- Threshold equal to 0 means all of channels will be multiplied.
- Threshold below 1 means top `(threshold * channels)` th of channels will be multiplied.
- Threshold above or equal to 1 means top `threshold` th of channels will be multiplied.

If you omit threshold, it will be interpreted as 0.
Note that even for same key, number of channels may vary. e.g. clip_l and clip_g has different number of channels. See "Tipical number of channels" below.

There are 10 types of option:
- "b": Threshold will be applied in top of range of `[:threshold]` and `[-value:]`.
- "o": Opposite of "b" will be applied.
- "m": Threshold will be applied in top of range of `[median-threshold:median+value]` where median is half of channels.
Note: With "b", "i", "m" option, threshold of 0 means not to apply emphasis to top of channels.
- "r": Threshold will be applied in top of range of `[threshold:value]`.
Note: If you specified "b", "f", "m", "r" option and no given value, threshold will be used as value.
- "c": Instead of emphasize top `threshold` th of channels, unsorted .range of channels of `[threshold:value]` will be emphasized.
Note: If you specified "c" option and no given value, value will be `threshold+1`.
- "n": Emphasis will be applied only at `value` th appearance of cross attention in each sampling.
Note: This option does not check out of bounds.
Important: Do not combine above 5 options. Combined behavior is undefined.
- "pa": Adds `value` before appling emphasis.
- "a": Adds `value` after appling emphasis.
- "ps": Subtracts `value` before appling emphasis.
- "s": Subtracts `value` after appling emphasis.
## Notes
### Typical number of channels
- "c": 768 for clip_l and 1280 for clip_g
- "k": out_features of to_k (for SDXL: 640 or 1280)
- "v": out_features of to_v (for SDXL: 640 or 1280)
- "q": For SDXL, for IN04, IN05, OUT03, OUT04, OUT05: `ceil(width_in_px/16)*ceil(height_in_px/16)*number_of_heads` where number_of_heads is 10.
If image size is 1920x1080, number of channels will be 81600.
For IN07, IN08, M00, OUT00, OUT01, OUT02: `ceil(width_in_px/32)*ceil(height_in_px/32)*number_of_heads` where number_of_heads is 20.
If image size is 1920x1080, number of channels will  be 40800.
- "s": Same as "q".