const API_BASE = '';

const cropSelect = document.getElementById('crop');
const countrySelect = document.getElementById('country');
const form = document.getElementById('predictForm');
const submitBtn = document.getElementById('submitBtn');
const resultBox = document.getElementById('result');
const resultNumber = document.getElementById('resultNumber');
const resultMeta = document.getElementById('resultMeta');
const errorBox = document.getElementById('errorBox');

fetch('/api/session').then(r => r.json()).then(d => {
    if (!d.logged_in) window.location.href = '/';
});

fetch(`${API_BASE}/crops`)
    .then(res => res.json())
    .then(crops => {
        crops.forEach(crop => {
            const opt = document.createElement('option');
            opt.value = crop;
            opt.textContent = crop;
            cropSelect.appendChild(opt);
        });
    })
    .catch(() => showError('Could not load crop list.'));

fetch(`${API_BASE}/countries`)
    .then(res => res.json())
    .then(countries => {
        countries.forEach(country => {
            const opt = document.createElement('option');
            opt.value = country;
            opt.textContent = country;
            countrySelect.appendChild(opt);
        });
    })
    .catch(() => showError('Could not load country list.'));

form.addEventListener('submit', function (e) {
    e.preventDefault();

    resultBox.classList.add('hidden');
    errorBox.classList.add('hidden');
    submitBtn.disabled = true;
    submitBtn.querySelector('span').textContent = 'Predicting…';

    const payload = {
        crop: cropSelect.value,
        country: countrySelect.value,
        year: document.getElementById('year').value,
        rainfall: document.getElementById('rainfall').value,
        temperature: document.getElementById('temperature').value,
        pesticide: document.getElementById('pesticide').value,
        area_harvested: document.getElementById('area').value
    };

    fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                showError(data.error);
            } else {
                resultNumber.textContent = data.predicted_yield_hg_ha.toLocaleString();
                resultMeta.textContent = `${data.crop} · ${data.country} · ${data.year}`;
                resultBox.classList.remove('hidden');
            }
        })
        .catch(() => showError('Could not reach the prediction server.'))
        .finally(() => {
            submitBtn.disabled = false;
            submitBtn.querySelector('span').textContent = 'Generate Prediction';
        });
});

function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove('hidden');
}