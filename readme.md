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
- "s" means is similar to "q" but applied after softmax.

Note that using "q", "s" requires to enable Extra Mode which makes slower because of disabling optimizations.

Threshold has 3 ways of interpretations:
- Threshold equal to 0 means all of channels will be multiplied.
- Threshold below 1 means top `(threshold * channels)` th of channels will be multiplied.
- Threshold above or equal to 1 means top threshold th of channels will be multiplied.