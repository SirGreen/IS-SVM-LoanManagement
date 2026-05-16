<!-- DASHBOARD -->
<div class="container py-5" style="background-color: #f8f9fa; min-height: 100vh; border-radius: 15px;">
    
    <div class="d-flex justify-content-between align-items-center mb-5">
        <div>
            <h2 class="fw-bold mb-0">Employer Dashboard</h2>
            <p class="text-muted">Manage loan applications and reviews</p>
        </div>
        <a href="index.php?page=employer&action=loan" class="btn btn-primary btn-lg rounded-pill px-4 shadow-sm fw-bold">
            <i class="bi bi-plus-lg me-2"></i>Make a Loan
        </a>
    </div>

    <div class="row g-4 mb-5">
            <div class="card border-0 shadow-sm p-4 h-100" style="border-left: 5px solid #ffc107 !important;">
                <div class="d-flex align-items-center">
                    <div class="rounded-circle bg-warning bg-opacity-10 p-3 me-3">
                        <span class="fs-3">⏳</span>
                    </div>
                    <div>
                        <h6 class="text-muted mb-1 text-uppercase small fw-bold">Pending Reviews</h6>
                        <h2 class="fw-bold mb-0"><?= $total_rewiew ?></h2>
                    </div>
                </div>
            </div>
    </div>

    <div class="card border-0 shadow-sm p-4">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h4 class="fw-bold mb-0">Pending Reviews List</h4>
            <span class="badge bg-light text-dark border fw-normal"><?= $total_rewiew ?> files waiting</span>
        </div>
        
        <div class="table-responsive">
            <table class="table table-hover align-middle">
                <thead class="table-light">
                    <tr>
                        <th class="border-0">Application ID</th>
                        <th class="border-0">Amount</th>
                        <th class="border-0">Date Received</th>
                        <th class="border-0">Decision Score</th>
                        <th class="border-0 text-end">Review</th>
                    </tr>
                </thead>
                <tbody>
                    <?php foreach ($pending_list['items'] as $item): ?>
                    <tr>
                        <td class="fw-bold text-primary"><?= $item['id'] ?></td>
                        <td class="fw-semibold"><?= $item['input']['credit_amount'] ?></td>
                        <td class="text-muted small"><?= date("d/m/Y", $item['timestamp']) ?></td>
                        <td>
                            <span class="badge rounded-pill bg-warning text-dark px-3 shadow-sm">
                                <?= round($item['decision_score'], 2) ?>
                            </span>
                        </td>
                        <td class="text-end">
                            <button  data-bs-toggle="modal" data-bs-target="#add-modal" data-modal="add-modal" data-review-id="<?=$item['id']?>"
                            class="review-btn btn btn-outline-dark btn-sm rounded-pill px-3 fw-bold">Review File</button>
                        </td>
                    </tr>
                    <?php endforeach; ?>
                </tbody>
            </table>
        </div>
        
        <div class="d-flex justify-content-center mt-3">
            <?php if ($total_pages > 1): ?>
            <nav class="mt-5">
                <ul class="pagination justify-content-center">
                    <!-- Previous -->
                    <li class="page-item <?= ($current_page <= 1) ? 'disabled' : '' ?>">
                        <a class="page-link" href="index.php?page=employer&action=dashboard&p=<?= $currentpage-1?>">Previous</a>
                    </li>

                    <!-- List -->
                    <?php for($i = 1; $i <= $total_pages; $i++): ?>
                        <li class="page-item <?= ($current_page == $i) ? 'active' : '' ?>">
                            <a class="page-link" href="index.php?page=employer&action=dashboard&p=<?= $i?>"><?= $i ?></a>
                        </li>
                    <?php endfor; ?>

                    <!-- Next -->
                    <li class="page-item <?= ($current_page >= $total_pages) ? 'disabled' : '' ?>">
                        <a class="page-link" href="index.php?page=employer&action=dashboard&p=<?= $currentpage+1?>">Next</a>
                    </li>
                </ul>
            </nav>
            <?php endif; ?>
        </div>
    </div>
</div>

<!-- MODAL -->
<div class="modal fade" id="add-modal" tabindex="-1" aria-labelledby="modalTitle" aria-hidden="true">
    <div class="modal-dialog modal-lg modal-dialog-centered"> <div class="modal-content p-4 border-0 shadow-lg">
            
            <div class="row g-3 mb-4 p-3">
                <div class="col-md-6 border-end">
                    <h5 class="fw-bold mb-3">Report Information</h5>
                    
                    <p class="mb-1 text-muted small ">Application ID</p>
                    <p id="modal-app-id" class="fw-bold fs-5"></p>

                    <p class="mb-1 text-muted small">Model Prediction</p>
                    <div class="mb-3">
                        <span id="modal-app-prediction" class="badge px-3 py-2"></span>
                    </div>

                    <p class="mb-1 text-muted small">Model Confidence</p>
                    <p id="modal-app-score" class="fw-bold fs-5"></p>

                    <p class="mb-1 text-muted small">Reported Date</p>
                    <p id="modal-app-timestamp" class="text-secondary"></p>
                </div>

                <div class="col-md-6 ps-md-4">
                    <h5 class="fw-bold mb-3">Application Details</h5>
                    <div class="row">
                        <div class="col-6 mb-3">
                            <small class="text-muted d-block">Existing Account</small>
                            <p id="modal-app-account"></p>
                        </div>
                        <div class="col-6 mb-3">
                            <small class="text-muted d-block">Credit History</small>
                            <p id="modal-app-history"></p>
                        </div>
                        <div class="col-6 mb-3">
                            <small class="text-muted d-block">Savings Account</small>
                            <p id="modal-app-savings"></p>
                        </div>
                        <div class="col-6 mb-3">
                            <small class="text-muted d-block">Age</small>
                            <p id="modal-app-age">-</p>
                        </div>
                        <div class="col-6 mb-3">
                            <small class="text-muted d-block">Duration</small>
                            <p id="modal-app-duration"></p>
                        </div>
                        <div class="col-6 mb-3">
                            <small class="text-muted d-block">Amount</small>
                            <p id="modal-app-amount"></p>
                        </div>
                        <div class="col-6">
                            <small class="text-muted d-block">Installment Rate</small>
                            <p id="modal-app-installement"></p>
                        </div>
                        <div class="col-6">
                            <small class="text-muted d-block">Purpose</small>
                            <p id="modal-app-purpose"></p>
                        </div>
                        <div class="col-6">
                            <small class="text-muted d-block">Present Employment since</small>
                            <p id="modal-app-employment"></p>
                        </div>
                        <div class="col-6">
                            <small class="text-muted d-block">Personal Status and sex</small>
                            <p id="modal-app-status"></p>
                        </div>
                        <div class="col-6">
                            <small class="text-muted d-block">Other debtors/garuantors</small>
                            <p id="modal-app-other"></p>
                        </div>
                        <div class="col-6">
                            <small class="text-muted d-block">Present residence since</small>
                            <p id="modal-app-residence"></p>
                        </div>
                        <div class="col-6">
                            <small class="text-muted d-block">Property</small>
                            <p id="modal-app-property"></p>
                        </div>
                        <div class="col-6">
                            <small class="text-muted d-block">Other installement plans</small>
                            <p id="modal-app-otherInstallement"></p>
                        </div>
                        <div class="col-6">
                            <small class="text-muted d-block">Housing</small>
                            <p id="modal-app-housing"></p>
                        </div>
                        <div class="col-6">
                            <small class="text-muted d-block">Number of existing credits at this bank</small>
                            <p id="modal-app-number"></p>
                        </div>
                        <div class="col-6">
                            <small class="text-muted d-block">Job</small>
                            <p id="modal-app-job"></p>
                        </div>
                        <div class="col-6">
                            <small class="text-muted d-block">Number of people being liable to provide maintenance for</small>
                            <p id="modal-app-liable"></p>
                        </div>
                        <div class="col-6">
                            <small class="text-muted d-block">Telephone</small>
                            <p id="modal-app-telephone"></p>
                        </div>
                        <div class="col-6">
                            <small class="text-muted d-block">Foreign Worker</small>
                            <p id="modal-app-foreign"></p>
                        </div>
                    </div>
                </div>
            </div>

            <div class="d-flex justify-content-end gap-2 p-2">
                <button type="button" class="btn btn-secondary px-4 py-2">Cancel</button>
                <button type="button" class="btn btn-danger px-4 py-2">
                    <i class="bi bi-x-circle me-1"></i> Reject Report
                </button>
                <button type="button" class="btn btn-success px-4 py-2">
                    <i class="bi bi-check2-circle me-1"></i> Approve Correction
                </button>
            </div>

        </div>
    </div>
</div>

<script>
document.addEventListener('DOMContentLoaded', function () {
    // 1. On récupère la liste PHP encodée en JSON directement dans le JS
    const pendingItems = <?= json_encode($pending_list['items']); ?>;
    
    const reviewButtons = document.querySelectorAll('.review-btn');
    
    reviewButtons.forEach(button => {
        button.addEventListener('click', function () {
            // Récupère l'ID embarqué dans le bouton
            const reviewId = this.getAttribute('data-review-id');
            
            // Trouve les données correspondantes dans notre tableau d'items
            const fileData = pendingItems.find(item => item.id == reviewId);
            
            if (fileData) {
                document.getElementById('modal-app-id').innerText = `#${fileData.id}`;
                document.getElementById('modal-app-amount').innerText = `${fileData.input.credit_amount}$`;
                
                const fileDate = new Date(fileData.timestamp * 1000);
                document.getElementById('modal-app-timestamp').innerText = fileDate.toLocaleString();
                
                const predictionSpan = document.getElementById('modal-app-prediction');
                if (fileData.prediction === 1) {
                    predictionSpan.innerText = 'APPROVE';
                    predictionSpan.className = 'badge px-3 py-2 bg-success';
                } else {
                    predictionSpan.innerText = 'REJECT';
                    predictionSpan.className = 'badge px-3 py-2 bg-danger';
                }
                const score = parseFloat(fileData.decision_score);
                const margin = parseFloat(fileData.uncertainty_margin);

                const k = margin > 0 ? (1.0 / margin) : 1.0; 
                const rawConfidence = 1.0 / (1.0 + Math.exp(-k * score));
                let confidence = 0;
                if (score >= 0) {
                    confidence = rawConfidence;
                } else {
                    confidence = 1.0 - rawConfidence;
                }
                const confidencePercentage = (confidence * 100).toFixed(1);
                document.getElementById('modal-app-score').innerText = `${confidencePercentage}%`
                
                document.getElementById('modal-app-age').innerText = fileData.input.age || 'N/A';
                document.getElementById('modal-app-installement').innerText = fileData.input.installment_rate_pct || 'N/A';
                document.getElementById('modal-app-duration').innerText = `${fileData.input.duration_months || 'N/A'} months`;
                document.getElementById('modal-app-history').innerText = fileData.input.credit_history || 'N/A';
                document.getElementById('modal-app-savings').innerText = fileData.input.savings_account || 'N/A';
                document.getElementById('modal-app-purpose').innerText = fileData.input.purpose || 'N/A';
                document.getElementById('modal-app-account').innerText = fileData.input.status_checking_account || 'N/A';
                document.getElementById('modal-app-employment').innerText = fileData.input.employment_since || 'N/A';
                document.getElementById('modal-app-status').innerText = fileData.input.personal_status_sex || 'N/A';
                document.getElementById('modal-app-other').innerText = fileData.input.other_debtors || 'N/A';
                document.getElementById('modal-app-residence').innerText = fileData.input.residence_since || 'N/A';
                document.getElementById('modal-app-property').innerText = fileData.input.property || 'N/A';
                document.getElementById('modal-app-otherInstallement').innerText = fileData.input.other_installment_plans || 'N/A';
                document.getElementById('modal-app-housing').innerText = fileData.input.housing || 'N/A';
                document.getElementById('modal-app-number').innerText = fileData.input.existing_credits || 'N/A';
                document.getElementById('modal-app-job').innerText = fileData.input.job || 'N/A';
                document.getElementById('modal-app-liable').innerText = fileData.input.liable_dependents || 'N/A';
                document.getElementById('modal-app-telephone').innerText = fileData.input.telephone || 'N/A';
                document.getElementById('modal-app-foreign').innerText = fileData.input.foreign_worker || 'N/A';

            }
        });
    });
});
</script>