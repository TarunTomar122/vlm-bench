# SmolVLM2 K4 Conclusions

## Scope

This completed replication evaluates identity-skipping four of the 27 SmolVLM2-2.2B vision
transformer blocks. It compares an evolved generic K4 route with five evolved capability routes,
matched-K baselines, a sealed IIIT5K OCR transfer set, and same-VM latency. No fine-tuning was
performed.

## Main Results

- Full-model accuracy on the 876-example method-selection split was 82.65%.
- The evolved generic K4 route `[4, 22, 24, 25]` reached 72.49%, a 10.16 percentage-point drop.
- The evolved capability-conditional policy reached 73.29%, only +0.80 pp over the generic route;
  its paired 95% interval was [-1.94, 3.54], so this does not confirm an overall conditional-route
  advantage.
- The generic evolved route beat the generic independent ranking by +4.91 pp [1.83, 7.99], and the
  evolved task policy beat the task-independent policy by +8.90 pp [5.82, 11.99]. Evolutionary
  route search therefore improves substantially on the naive single-block-ranking baselines.
- Conditional routes helped counting (+7.18 pp [1.10, 13.26]) and spatial (+9.39 pp [2.76, 16.02])
  relative to generic K4, but hurt OCR (-13.55 pp [-21.94, -5.16]).

## External OCR Result

On the sealed 250-example IIIT5K word-OCR set, the full model achieved 94.8%, generic K4 achieved
86.4%, and the OCR-specific K4 route achieved 72.8%. The OCR route was -13.60 pp below generic K4
with paired 95% interval [-19.20, -8.40]. The internal OCR-specific route therefore did not
transfer to this external OCR source.

## Efficiency Result

The generic K4 route skips 60,958,016 vision parameters: 14.76% of the vision encoder and 2.71% of
the 2.247B-parameter model. On 50 balanced development examples with five repeats, vision-encoder
median latency improved 8.60% and end-to-end median latency improved 4.19%.

The cloud provider denied GPU clock locking, so this is an `unlocked_same_vm_fallback` measurement,
not a fixed-clock or edge-device latency claim.

## K6 Diagnostic

K6 was stopped before matched controls or external transfer because its generic route lost 21.80 pp
overall on the internal selection data. Its partial artifacts are diagnostic-only, not completed
evidence.

## Conclusion

The supported claim is modest: model-specific evolutionary search can find a generic four-block
vision route that is materially better than naive and random K4 pruning, with a measurable latency
gain. This experiment does not support a general task-conditional routing benefit or an OCR-specific
route that transfers across OCR sources.
