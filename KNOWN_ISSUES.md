# Known Issues

## Visualization (`viz2psy-viz`)

### Saliency maps display smaller than source images in browse mode

**Location**: `viz2psy-viz image --browse` with saliency panel

**Description**: When using the browsable viewer with the saliency panel, saliency heatmaps display at a smaller size than the source images shown in the left panel. The aspect ratio is correct, but the overall size is reduced.

**Cause**: Saliency images are rendered at a fixed height (300px) to maintain consistent y-axis range across frames when using Plotly's animation/slider mechanism. Without this constraint, only the top half of the saliency map would render.

**Workaround**: Use the single-image viewer (`--no-browse`) for full-size saliency display, or view individual frames without the slider.

**Potential fix**: Investigate Plotly layout image positioning (similar to the source image) instead of go.Image trace for saliency display, which would allow independent sizing.

---

### Saliency model uses CPU on Apple Silicon (MPS)

**Location**: `viz2psy.models.saliency.SaliencyModel`

**Description**: On Apple Silicon Macs, the saliency model runs on CPU instead of MPS GPU acceleration, resulting in slower inference.

**Cause**: DeepGaze IIE uses float64 tensors internally, which MPS does not support. The model automatically falls back to CPU with a warning.

**Workaround**: None currently. Use CUDA GPU if available for faster saliency inference.
