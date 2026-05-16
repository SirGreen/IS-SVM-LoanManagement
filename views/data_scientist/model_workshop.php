<!-- ====== FLASH ====== -->
<?php 
if (isset($_SESSION['flash'])){ 

  $flashType = $_SESSION['flash_type'] ?? 'success';
  $isDanger = $flashType === 'danger';
  $bgColor  = $isDanger ? "#f8d7da" : "#d4edda";
  $brColor  = $isDanger ? "#f5c6cb" : "#c3e6cb";
  $txtColor = $isDanger ? "#721c24" : "#155724";
}
if (isset($_SESSION['flash'])): ?>
  <div class="alert" style="padding: 15px; background-color: <?= $bgColor ?>; color: <?= $txtColor ?>; border: 1px solid <?= $brColor ?>; border-radius: 4px; margin-bottom: 20px;">
    <?= htmlspecialchars($_SESSION['flash']); ?>
  </div>
    <?php 
        unset($_SESSION['flash']); 
    ?>
<?php endif; ?>

<!-- ====== WORKSHOP ====== -->
<div class="container-fluid py-4" style="background-color: #e9ecef; min-height: 100vh; font-family: sans-serif;">
    
     <!-- Select Model -->
    <div class="d-flex justify-content-between align-items-center mb-4">
        <h2 class="fw-bold">Model Management Dashboard</h2>
        
        <!-- Conteneur du menu déroulant -->
        <div class="dropdown">
            <button class="btn btn-secondary dropdown-toggle rounded-pill px-3 py-2" 
                    type="button" 
                    id="modelDropdown" 
                    data-bs-toggle="dropdown" 
                    aria-expanded="false">
                Current model : SVM_v1
            </button>
            <ul class="dropdown-menu dropdown-menu-end" aria-labelledby="modelDropdown">
                <li><h6 class="dropdown-header">Select a version</h6></li>
                <li><a class="dropdown-item active" href="#">SVM_v1 (Current)</a></li>
                <li><a class="dropdown-item disabled" href="#">SVM_v2</a></li>
            </ul>
        </div>
    </div>

    <!-- Stats -->
    <div class="row g-4">
        <div class="col-md-6">
            <div class="card border-0 shadow-sm h-100 p-4">
                <h5 class="fw-bold mb-4">Confusion Matrix (Validation)</h5>
                <div class="row text-center mb-2">
                    <div class="col-4 offset-4 small text-muted">Actual : Good</div>
                    <div class="col-4 small text-muted">Actual : Bad</div>
                </div>
                <div class="row align-items-center text-center g-2">
                    <div class="col-4 fw-bold small text-muted text-uppercase">Prediction</div>
                    <div class="col-4">
                        <div class="bg-success text-white fw-bold py-3 rounded"><?= $stats['confusion_matrix']['true_positive']?></div>
                    </div>
                    <div class="col-4">
                        <div class="bg-secondary text-white fw-bold py-3 rounded" style="opacity: 0.6;"><?= $stats['confusion_matrix']['false_positive']?></div>
                    </div>
                    <div class="col-4 offset-4">
                        <div class="bg-secondary text-white fw-bold py-3 rounded" style="opacity: 0.6;"><?= $stats['confusion_matrix']['false_negative']?></div>
                    </div>
                    <div class="col-4">
                        <div class="bg-success text-white fw-bold py-3 rounded"><?= $stats['confusion_matrix']['true_negative']?></div>
                    </div>
                </div>
            </div>
        </div>

        <div class="col-md-6">
            <div class="card border-0 shadow-sm h-100 p-4 text-center">
                <h5 class="fw-bold mb-4 text-start">Performance KPI</h5>
                <div class="row mt-3">
                    <div class="col-4">
                        <h2 class="text-primary fw-bold mb-0"><?= round($stats['accuracy'],2)?>%</h2>
                        <small class="text-muted">Accuracy</small>
                    </div>
                    <div class="col-4">
                        <h2 class="text-primary fw-bold mb-0"><?= round($stats['f1_score'],2)?>%</h2>
                        <small class="text-muted">F1-Score</small>
                    </div>
                    <div class="col-4">
                        <h2 class="text-primary fw-bold mb-0"><?= round($stats['recall'],2)?>%</h2>
                        <small class="text-muted">Recall</small>
                    </div>
                </div>
                <div class="mt-4 p-2 bg-light border rounded small">
                    Benchmark : Within 0.8% of Scikit-Learn reference.
                </div>
            </div>
        </div>

        <div class="col-md-6">
            <div class="card border-0 shadow-sm p-4 h-100">
                <h5 class="fw-bold mb-3">Hyperplan & Margin visualisation</h5>
                <div class="bg-light rounded fw-bold d-flex align-items-center justify-content-center mb-3" style="height: 200px; border: 1px dashed #ccc;">
                    <span class="text-muted">Current Uncertainty Margin : <?= $stats['uncertainty_margin']?></span>
                </div>
                <form action="index.php?page=model_workshop&action=update_margin" method="POST" class="d-flex gap-2 align-items-center">
                    <input type="number" step="0.05" name="margin" id="margin" class="form-control form-control-sm bg-light" value="<?=$stats['uncertainty_margin']?>" required>
                    <button type="submit" class="btn btn-light btn-sm border rounded-pill text-nowrap">Update margin</button>
                </form>
            </div>
        </div>

        <div class="col-md-6">
            <div class="card border-0 shadow-sm p-4 h-100">
                <div class="d-flex justify-content-between">
                    <h5 class="fw-bold">File used for training datasets</h5>
                </div>
                <div class="bg-light rounded fw-bold d-flex align-items-center justify-content-center mb-3" style="height: 200px; border: 1px dashed #ccc;">
                    <span class="text-muted"><?= $stats['dataset_zip_path']?></span>
                </div>
            </div>
        </div>

        <div class="col-12">
            <div class="card border-warning shadow-sm p-4" style="border: 2px solid #fd7e14 !important; background-color: #fffaf5;">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <div class="d-flex align-items-center">
                        <div class="bg-orange p-2 rounded me-2 text-white" style="background-color: #fd7e14;">$</div>
                        <div>
                            <h5 class="fw-bold mb-0">Asymmetric Misclassification Cost Tracker</h5>
                            <small class="text-muted">Real-time Financial Risk Exposure Monitor (5:1 Cost Ratio)</small>
                        </div>
                    </div>
                    <span class="badge bg-orange rounded-pill" style="background-color: #fd7e14;">LIVE</span>
                </div>

                <div class="row g-3">
                    <div class="col-md-4">
                        <div class="card border-danger border-2 p-3">
                            <small class="text-danger fw-bold"><i class="bi bi-exclamation-triangle"></i> Total Misclassification Cost</small>
                            <h1 class="fw-bold my-2" style="color: #d9534f;"><?= $stats['confusion_matrix']['false_positive']*5 + $stats['confusion_matrix']['false_negative']?></h1>
                            <small class="text-muted d-block mb-2">Cost Units (FP×5 + FN×1)</small>
                            <div class="d-flex justify-content-between small"><span>False Positives:</span><span class="fw-bold text-danger"><?= $stats['confusion_matrix']['false_positive']?> × 5 = <?= $stats['confusion_matrix']['false_positive']*5?></span></div>
                            <div class="d-flex justify-content-between small"><span>False Negatives:</span><span class="fw-bold text-danger"><?= $stats['confusion_matrix']['false_negative']?> × 1 = <?= $stats['confusion_matrix']['false_negative']?></span></div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card border-primary border-2 p-3">
                            <small class="text-primary fw-bold">Cost-Weighted Accuracy</small>
                            <h1 class="fw-bold my-2" style="color: #4e73df;"><?= round((1- ($stats['confusion_matrix']['false_positive']*5 + $stats['confusion_matrix']['false_negative']*1)/(($stats['confusion_matrix']['true_negative']+$stats['confusion_matrix']['false_positive'])*5 + ($stats['confusion_matrix']['true_positive']+$stats['confusion_matrix']['false_negative'])*1))*100,2)?>%</h1>
                            <small class="text-muted">Financial performance metric</small>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card border-success border-2 p-3">
                            <small class="text-success fw-bold">$ Avg. Cost per Prediction</small>
                            <h1 class="fw-bold my-2" style="color: #198754;">0.14</h1>
                            <small class="text-muted">Cost units per decision</small>
                            <hr>
                            <div class="d-flex justify-content-between small"><span>Total Predictions:</span><span class="fw-bold"><?= $stats['confusion_matrix']['true_negative'] + $stats['confusion_matrix']['true_positive'] + $stats['confusion_matrix']['false_negative'] + $stats['confusion_matrix']['false_positive']?></span></div>
                        </div>
                    </div>
                </div>
                
                <div class="mt-3 p-2 bg-warning bg-opacity-10 border border-warning rounded small">
                    <i class="bi bi-exclamation-circle"></i> <strong>Cost Matrix Configuration (5:1 Penalty Ratio)</strong><br>
                    False Positive (Approve Bad Loan): Cost = 5 units | False Negative (Reject Good Loan): Cost = 1 unit
                </div>
            </div>
        </div>

        <div class="col-12">
            <div class="card border-0 shadow-sm p-4">
                <h5 class="fw-bold mb-4">Train New SVM Model Instance</h5>
                <form class="row g-3" action="index.php?page=model_workshop&action=train" method="POST" enctype="multipart/form-data">
                    <div class="col-md-6">
                        <label for="dataset" class="form-label fw-bold small">Dataset (.zip)</label>
                        <input type="file" class="form-control" id="dataset" name="dataset" accept=".zip" required>
                    </div>
                    <div class="col-md-3">
                        <label class="form-label fw-bold small">Hyperparameter C</label>
                        <input type="number" name="hyperparameter" id="hyparamater" class="form-control form-control-sm bg-light" required>
                    </div>
                    <div class="col-md-3">
                        <label class="form-label fw-bold small">Epochs</label>
                        <input type="number" name="epochs" id="epochs" class="form-control form-control-sm bg-light" required>
                    </div>
                    <div class="col-12">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="confirmTrain">
                            <label class="form-check-label small" for="confirmTrain">I confirm that both training and test datasets are properly formatted</label>
                        </div>
                    </div>

                    <button type="submit" class="btn btn-secondary px-5">Start</button>
                </form>
            </div>
        </div>
    </div>
</div>