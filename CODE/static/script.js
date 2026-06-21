document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const btnBrowse = document.getElementById('btn-browse');
    const uploadSection = document.getElementById('upload-section');
    const processingSection = document.getElementById('processing-section');
    const resultsSection = document.getElementById('results-section');
    
    const visionImage = document.getElementById('vision-image');
    const visionInfo = document.getElementById('vision-info');
    const digitalBoard = document.getElementById('digital-board');
    const btnSolve = document.getElementById('btn-solve');
    const btnReset = document.getElementById('btn-reset');
    const solveInfo = document.getElementById('solve-info');
    
    const btnDebug = document.getElementById('btn-debug');
    const debugModal = document.getElementById('debug-modal');
    const closeModal = document.getElementById('close-modal');
    const debugGrid = document.getElementById('debug-grid');

    // Global State
    let currentGrid = [];
    let inputs = [];

    // Initialize Board
    function initBoard() {
        digitalBoard.innerHTML = '';
        inputs = [];
        for (let i = 0; i < 81; i++) {
            const input = document.createElement('input');
            input.type = 'text';
            input.maxLength = 1;
            input.className = 'sudoku-cell';
            input.dataset.index = i;
            
            // Allow only numbers 1-9
            input.addEventListener('input', (e) => {
                const val = e.target.value;
                if (!/^[1-9]$/.test(val)) {
                    e.target.value = '';
                }
            });

            // Grid navigation with arrows
            input.addEventListener('keydown', handleArrowNavigation);

            digitalBoard.appendChild(input);
            inputs.push(input);
        }
    }

    function handleArrowNavigation(e) {
        const index = parseInt(e.target.dataset.index);
        let newIndex = -1;

        if (e.key === 'ArrowRight') newIndex = index + 1;
        if (e.key === 'ArrowLeft') newIndex = index - 1;
        if (e.key === 'ArrowDown') newIndex = index + 9;
        if (e.key === 'ArrowUp') newIndex = index - 9;

        if (newIndex >= 0 && newIndex < 81) {
            e.preventDefault();
            inputs[newIndex].focus();
        }
    }

    // Upload Logic
    btnBrowse.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    async function handleFile(file) {
        if (!file.type.match('image.*')) {
            alert('Por favor, sube una imagen válida (JPG, PNG).');
            return;
        }

        uploadSection.classList.add('hidden');
        processingSection.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Error procesando la imagen');
            }

            populateResults(data);

        } catch (error) {
            alert(`Error: ${error.message}`);
            resetApp();
        }
    }

    function populateResults(data) {
        processingSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        // Set Vision Image
        visionImage.src = data.board_image;
        visionInfo.textContent = `Se detectó el tablero y ${data.corrections.length > 0 ? 'se aplicaron correcciones lógicas' : 'se leyeron los dígitos con alta confianza'}.`;
        visionInfo.className = `info-banner ${data.corrections.length > 0 ? 'error' : 'success'}`;

        // Set Digital Board
        currentGrid = data.grid;
        inputs.forEach((input, i) => {
            const row = Math.floor(i / 9);
            const col = i % 9;
            const val = currentGrid[row][col];
            
            input.value = val === 0 ? '' : val;
            input.className = 'sudoku-cell';
            if (val !== 0) {
                input.classList.add('detected');
            }
        });

        // Set Debug Modal
        debugGrid.innerHTML = '';
        data.cells_images.forEach((imgSrc, i) => {
            const row = Math.floor(i / 9);
            const col = i % 9;
            const cell = document.createElement('div');
            cell.className = 'debug-cell';
            
            const img = document.createElement('img');
            img.src = imgSrc;
            
            const span = document.createElement('span');
            span.textContent = currentGrid[row][col] === 0 ? '-' : currentGrid[row][col];
            
            cell.appendChild(img);
            cell.appendChild(span);
            debugGrid.appendChild(cell);
        });

        // Poblamos los Logs Matemáticos
        const debugLogsEl = document.getElementById('debug-logs');
        let logsString = "=== LOGS MATEMÁTICOS DE LA CNN (VERSIÓN 2 - AUTO-CONTRASTE) ===\n\n";
        if (data.logs && data.logs.length > 0) {
            data.logs.forEach(log => {
                logsString += `CASILLA ${log.casilla} (Fila ${log.fila}, Col ${log.col}):\n`;
                logsString += `  > Tensor Píxeles -> Min: ${log.pixel_min.toFixed(2)} | Max: ${log.pixel_max.toFixed(2)} | Media: ${log.pixel_media.toFixed(2)}\n`;
                
                // Formatear logits array (por ejemplo: [-1.09, -2.09, ...])
                const formattedLogits = log.logits.map(val => val.toFixed(2)).join(', ');
                logsString += `  > Logits Red     -> [${formattedLogits}]\n`;
                logsString += `  > Veredicto Final-> ${log.veredicto} (Confianza/Logit: ${log.confianza.toFixed(2)})\n`;
                logsString += "----------------------------------------\n";
            });
        } else {
            logsString += "No hay logs disponibles para esta detección.\n";
        }
        debugLogsEl.value = logsString;

        const btnCopyLogs = document.getElementById('btn-copy-logs');
        if (btnCopyLogs) {
            btnCopyLogs.onclick = async () => {
                try {
                    await navigator.clipboard.writeText(logsString);
                    const originalText = btnCopyLogs.textContent;
                    btnCopyLogs.textContent = '✅ ¡Copiado!';
                    setTimeout(() => {
                        btnCopyLogs.textContent = originalText;
                    }, 2000);
                } catch (err) {
                    console.error('Failed to copy: ', err);
                }
            };
        }

        solveInfo.classList.add('hidden');
    }

    // Solve Logic
    btnSolve.addEventListener('click', async () => {
        // Build grid from inputs
        const grid = [];
        for (let r = 0; r < 9; r++) {
            const row = [];
            for (let c = 0; c < 9; c++) {
                const val = inputs[r * 9 + c].value;
                row.push(val === '' ? 0 : parseInt(val));
            }
            grid.push(row);
        }

        btnSolve.disabled = true;
        btnSolve.textContent = 'Resolviendo...';

        try {
            const response = await fetch('/api/solve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ grid })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Error resolviendo el Sudoku');
            }

            // Animate Solution
            const solvedGrid = data.grid;
            for (let r = 0; r < 9; r++) {
                for (let c = 0; c < 9; c++) {
                    const idx = r * 9 + c;
                    if (inputs[idx].value === '') {
                        inputs[idx].value = solvedGrid[r][c];
                        inputs[idx].classList.add('solved');
                    }
                }
            }

            solveInfo.textContent = '¡Sudoku resuelto con éxito!';
            solveInfo.className = 'info-banner success';
            solveInfo.classList.remove('hidden');

        } catch (error) {
            solveInfo.textContent = error.message;
            solveInfo.className = 'info-banner error';
            solveInfo.classList.remove('hidden');
        } finally {
            btnSolve.disabled = false;
            btnSolve.textContent = 'Resolver Sudoku';
        }
    });

    // Reset Logic
    function resetApp() {
        uploadSection.classList.remove('hidden');
        processingSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        fileInput.value = '';
        initBoard();
    }

    btnReset.addEventListener('click', resetApp);

    // Modal Logic
    btnDebug.addEventListener('click', () => debugModal.classList.remove('hidden'));
    closeModal.addEventListener('click', () => debugModal.classList.add('hidden'));
    debugModal.addEventListener('click', (e) => {
        if (e.target === debugModal) debugModal.classList.add('hidden');
    });

    // Initialize
    initBoard();
});
