#!/bin/bash
# Generate example images for README documentation
# Run from project root: bash docs/images/generate_examples.sh

set -e
cd "$(dirname "$0")/../.."

IMAGES_DIR="docs/images"
DATA_DIR="data"

echo "Generating example visualizations..."
echo "Output directory: $IMAGES_DIR"
echo ""

# 1. Dashboard screenshot (static - just note that dashboard HTML exists)
echo "Note: Dashboard is interactive HTML. Take a screenshot manually from:"
echo "  $DATA_DIR/shared1000_scores_dashboard.html"
echo ""

# 2. Single image viewer with feature panels (browsable)
echo "[1/6] Generating browsable image viewer..."
viz2psy-viz image "$DATA_DIR/shared1000_scores.csv" \
    --image-root "$DATA_DIR/shared1000" \
    --browse \
    --max-rows 50 \
    --panels emotions_bar,scalars,places \
    -o "$IMAGES_DIR/example_image_viewer.html"

# 3. Composite visualization (static PNG)
echo "[2/6] Generating composite visualization..."
viz2psy-viz composite \
    "$DATA_DIR/shared1000/shared0001_nsd02951.png" \
    "$DATA_DIR/shared1000_scores.csv" \
    --panels saliency,emotions,bars \
    -o "$IMAGES_DIR/example_composite.png"

# 4. Scatter plot - CLIP embeddings with UMAP
echo "[3/6] Generating CLIP scatter plot..."
viz2psy-viz scatter "$DATA_DIR/shared1000_scores.csv" \
    --features "clip_*" \
    --method umap \
    --color-by aesthetics \
    -i \
    -o "$IMAGES_DIR/example_scatter_clip.html"

# 5. Emotion scatter plot
echo "[4/6] Generating emotion scatter plot..."
viz2psy-viz scatter "$DATA_DIR/shared1000_scores.csv" \
    --features Adoration Amusement Anxiety Awe Disgust Excitement Fear Joy Sadness Surprise \
    --method mds \
    --color-by memorability \
    -i \
    -o "$IMAGES_DIR/example_scatter_emotions.html"

# 6. Timeseries for video data
echo "[5/6] Generating video timeseries..."
viz2psy-viz timeseries "$DATA_DIR/mr_bean_scores.csv" \
    --features memorability aesthetics \
    --show-diff \
    --rolling-window 10 \
    -i \
    -o "$IMAGES_DIR/example_timeseries.html"

# 7. Correlation heatmap
echo "[6/6] Generating correlation heatmap..."
viz2psy-viz heatmap "$DATA_DIR/shared1000_scores.csv" \
    --features memorability aesthetics colorfulness rms_contrast edge_density \
              Adoration Amusement Anxiety Awe Disgust Excitement Fear Joy Sadness Surprise \
    -o "$IMAGES_DIR/example_heatmap.png"

echo ""
echo "Done! Generated files in $IMAGES_DIR:"
ls -la "$IMAGES_DIR"

echo ""
echo "Next steps:"
echo "1. Open HTML files in browser and take screenshots"
echo "2. Save screenshots as PNG files with descriptive names"
echo "3. Add to README.md with relative paths like:"
echo '   ![Dashboard](docs/images/dashboard.png)'
