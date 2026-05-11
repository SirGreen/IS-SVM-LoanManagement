<div class="container-fluid">
    <h2 class="mb-4">Tableau de bord</h2>
    
    <div class="row mb-4">
        <div class="col-md-3">
            <div class="card shadow-sm border-0"><div class="card-body"><h5>Total</h5><h3>1,280</h3></div></div>
        </div>
        </div>

    <div class="card shadow-sm border-0">
        <div class="card-body">
            <canvas id="myChart" height="100"></canvas>
        </div>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
const ctx = document.getElementById('myChart');
new Chart(ctx, {
    type: 'line',
    data: {
        labels: ['Jan', 'Fev', 'Mar', 'Avr', 'Mai'],
        datasets: [{
            label: 'Performance',
            data: [12, 19, 3, 5, 2],
            borderColor: '#0d6efd',
            tension: 0.4
        }]
    }
});
</script>