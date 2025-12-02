// Import PDF.js ES module
import * as pdfjsLib from "./libs/pdf.min.mjs";

pdfjsLib.GlobalWorkerOptions.workerSrc = "./libs/pdf.worker.min.mjs";

const selectBtn = document.getElementById('select-pdf');
const fileInput = document.getElementById('pdf-upload');
const uploadBox = document.getElementById('upload-box');

// Open file dialog when button clicked
selectBtn.addEventListener('click', () => fileInput.click());

// Handle file selection
fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Show processing message
    let processingMsg = document.createElement('p');
    processingMsg.textContent = 'Processing PDF...';
    uploadBox.appendChild(processingMsg);

    const arrayBuffer = await file.arrayBuffer();

    try {
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        const page = await pdf.getPage(1);
        const viewport = page.getViewport({ scale: 2 });
        const canvas = document.createElement('canvas');
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        const context = canvas.getContext('2d');

        await page.render({ canvasContext: context, viewport: viewport }).promise;

        // Convert canvas to PNG
        const pngDataUrl = canvas.toDataURL('image/png');

        // Remove old download link & processing message
        const oldLink = uploadBox.querySelector('a');
        if (oldLink) oldLink.remove();
        processingMsg.remove();

        // Create download link
        const link = document.createElement('a');
        link.href = pngDataUrl;
        link.download = file.name.replace('.pdf', '.png');
        link.textContent = 'Download PNG';
        link.style.display = 'block';
        link.style.marginTop = '20px';
        uploadBox.appendChild(link);

    } catch (err) {
        processingMsg.textContent = 'Error processing PDF. Please try another file.';
        console.error(err);
    }
});

// Optional: Drag & drop
uploadBox.addEventListener('dragover', e => {
    e.preventDefault();
    uploadBox.style.background = '#e0f0ff';
});
uploadBox.addEventListener('dragleave', e => {
    e.preventDefault();
    uploadBox.style.background = '#f9fbff';
});
uploadBox.addEventListener('drop', e => {
    e.preventDefault();
    uploadBox.style.background = '#f9fbff';
    const file = e.dataTransfer.files[0];
    if (file && file.type === 'application/pdf') {
        fileInput.files = e.dataTransfer.files;
        fileInput.dispatchEvent(new Event('change'));
    }
});
