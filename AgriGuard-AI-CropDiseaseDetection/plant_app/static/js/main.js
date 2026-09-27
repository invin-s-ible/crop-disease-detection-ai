// AgriGuard AI - shared front-end behavior

document.addEventListener('DOMContentLoaded', function () {
  // Auto-dismiss server-rendered flash alerts after 6 seconds.
  // Scoped to .alert-dismissible only, so JS-controlled boxes like #uploadError
  // (which aren't dismissible flash messages) are never auto-removed before
  // they get a chance to display an error.
  document.querySelectorAll('.alert.alert-dismissible').forEach(function (alertEl) {
    setTimeout(function () {
      if (window.bootstrap) {
        var alert = window.bootstrap.Alert.getOrCreateInstance(alertEl);
        alert.close();
      }
    }, 6000);
  });

  setupImageUpload();
});

function setupImageUpload() {
  const dropzone = document.getElementById('uploadDropzone');
  const fileInput = document.getElementById('image_file');
  const preview = document.getElementById('imagePreview');
  const form = document.getElementById('imageUploadForm');
  const submitBtn = document.getElementById('analyzeBtn');
  const spinner = document.getElementById('analyzeSpinner');
  const errorBox = document.getElementById('uploadError');

  if (!dropzone || !fileInput || !form) return;

  dropzone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      showPreview(fileInput.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) showPreview(fileInput.files[0]);
  });

  function showPreview(file) {
    if (!file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      preview.src = e.target.result;
      preview.style.display = 'block';
    };
    reader.readAsDataURL(file);
  }

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    if (!fileInput.files.length) {
      showError('Please select or drop a leaf image first.');
      return;
    }

    const formData = new FormData(form);
    submitBtn.disabled = true;
    spinner.style.display = 'block';
    submitBtn.querySelector('span').textContent = 'Analyzing...';
    hideError();

    fetch(form.action, {
      method: 'POST',
      body: formData,
      headers: { 'X-Requested-With': 'XMLHttpRequest' }
    })
      .then(res => res.json().then(data => ({ status: res.status, data })))
      .then(({ status, data }) => {
        if (status === 200 && data.success) {
          window.location.href = data.redirect;
        } else {
          showError(data.error || 'Analysis failed. Please try again.');
          resetButton();
        }
      })
      .catch(() => {
        showError('Network error. Please check your connection and try again.');
        resetButton();
      });
  });

  function resetButton() {
    submitBtn.disabled = false;
    spinner.style.display = 'none';
    submitBtn.querySelector('span').textContent = 'Analyze Leaf Image';
  }

  function showError(msg) {
    if (!errorBox) return;
    errorBox.textContent = msg;
    errorBox.style.display = 'block';
  }
  function hideError() {
    if (!errorBox) return;
    errorBox.style.display = 'none';
  }
}
