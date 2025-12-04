// Example SAM-assisted segmentation function
// Attach to window so it's globally accessible
window.samSegment = async function(canvas, rect) {
    // rect = {x, y, width, height}
    // For now, this is a placeholder that draws a rectangle mask
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    // Draw semi-transparent overlay
    ctx.fillStyle = 'rgba(0, 0, 255, 0.3)';
    ctx.fillRect(rect.x, rect.y, rect.width, rect.height);

    // Return simulated mask data
    const mask = ctx.getImageData(rect.x, rect.y, rect.width, rect.height);
    return mask;
};
