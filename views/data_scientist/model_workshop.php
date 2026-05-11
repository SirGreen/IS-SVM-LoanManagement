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
                <li><a class="dropdown-item" href="#">SVM_v2</a></li>
                <li><a class="dropdown-item" href="#">SVM_v3</a></li>
            </ul>
        </div>
    </div>

    <!-- Stats -->
    <div class="row g-4">
        <div class="col-md-6">
            <div class="card border-0 shadow-sm h-100 p-4">
                <h5 class="fw-bold mb-4">Confusion Matrix (Validation)</h5>
                <div class="row text-center mb-2">
                    <div class="col-4 offset-4 small text-muted">Pred : Good</div>
                    <div class="col-4 small text-muted">Pred : Bad</div>
                </div>
                <div class="row align-items-center text-center g-2">
                    <div class="col-4 fw-bold small text-muted text-uppercase">Actual</div>
                    <div class="col-4">
                        <div class="bg-success text-white fw-bold py-3 rounded">190</div>
                    </div>
                    <div class="col-4">
                        <div class="bg-secondary text-white fw-bold py-3 rounded" style="opacity: 0.6;">12</div>
                    </div>
                    <div class="col-4 offset-4">
                        <div class="bg-secondary text-white fw-bold py-3 rounded" style="opacity: 0.6;">5</div>
                    </div>
                    <div class="col-4">
                        <div class="bg-success text-white fw-bold py-3 rounded">63</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="col-md-6">
            <div class="card border-0 shadow-sm h-100 p-4 text-center">
                <h5 class="fw-bold mb-4 text-start">Performance KPI</h5>
                <div class="row mt-3">
                    <div class="col-4">
                        <h2 class="text-primary fw-bold mb-0">86.4%</h2>
                        <small class="text-muted">Accuracy</small>
                    </div>
                    <div class="col-4">
                        <h2 class="text-primary fw-bold mb-0">76%</h2>
                        <small class="text-muted">F1-Score</small>
                    </div>
                    <div class="col-4">
                        <h2 class="text-primary fw-bold mb-0">65.4%</h2>
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
                <div class="bg-light rounded d-flex align-items-center justify-content-center mb-3" style="height: 200px; border: 1px dashed #ccc;">
                    <span class="text-muted">Visualization placeholder</span>
                </div>
                <button class="btn btn-light btn-sm border rounded-pill px-3 w-25">Update margin</button>
            </div>
        </div>

        <div class="col-md-6">
            <div class="card border-0 shadow-sm p-4 h-100">
                <div class="d-flex justify-content-between">
                    <h5 class="fw-bold">Training Loss</h5>
                    <small class="text-muted">Accuracy</small>
                </div>
                <div class="bg-light rounded d-flex align-items-center justify-content-center mb-3" style="height: 200px; border: 1px dashed #ccc;">
                    <span class="text-muted">Visualization placeholder</span>
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
                            <h1 class="fw-bold my-2" style="color: #d9534f;">37</h1>
                            <small class="text-muted d-block mb-2">Cost Units (FP×5 + FN×1)</small>
                            <div class="d-flex justify-content-between small"><span>False Positives:</span><span class="fw-bold text-danger">5 × 5 = 25</span></div>
                            <div class="d-flex justify-content-between small"><span>False Negatives:</span><span class="fw-bold text-danger">12 × 1 = 12</span></div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card border-primary border-2 p-3">
                            <small class="text-primary fw-bold">Cost-Weighted Accuracy</small>
                            <h1 class="fw-bold my-2" style="color: #4e73df;">93.2%</h1>
                            <small class="text-muted">Financial performance metric</small>
                            <div class="progress mt-3" style="height: 10px;">
                                <div class="progress-bar" style="width: 93%;"></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card border-success border-2 p-3">
                            <small class="text-success fw-bold">$ Avg. Cost per Prediction</small>
                            <h1 class="fw-bold my-2" style="color: #198754;">0.14</h1>
                            <small class="text-muted">Cost units per decision</small>
                            <hr>
                            <div class="d-flex justify-content-between small"><span>Total Predictions:</span><span class="fw-bold">270</span></div>
                        </div>
                    </div>
                </div>

                <div class="mt-4">
                    <h6 class="fw-bold small">Misclassification Cost Trend</h6>
                    <div class="d-flex align-items-end gap-2" style="height: 100px;">
                        <div class="bg-orange opacity-75 w-100" style="height: 90%; background-color: #fd7e14;"></div>
                        <div class="bg-orange opacity-75 w-100" style="height: 70%; background-color: #fd7e14;"></div>
                        <div class="bg-orange opacity-75 w-100" style="height: 50%; background-color: #fd7e14;"></div>
                        <div class="bg-orange opacity-75 w-100" style="height: 35%; background-color: #fd7e14;"></div>
                        <div class="bg-orange opacity-75 w-100" style="height: 25%; background-color: #fd7e14;"></div>
                        <div class="bg-orange opacity-75 w-100" style="height: 15%; background-color: #fd7e14;"></div>
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
                <form class="row g-3">
                    <div class="col-md-3">
                        <label class="form-label fw-bold small">Training Dataset (.csv)</label>
                        <div class="input-group">
                            <input type="text" class="form-control form-control-sm bg-light" readonly>
                            <button class="btn btn-outline-secondary btn-sm" type="button">Choose a file</button>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <label class="form-label fw-bold small">Hyperparameter C</label>
                        <input type="text" class="form-control form-control-sm bg-light" value="1.00">
                    </div>
                    <div class="col-md-3">
                        <label class="form-label fw-bold small">Epochs</label>
                        <input type="text" class="form-control form-control-sm bg-light" value="1000">
                    </div>
                    <div class="col-md-3">
                        <label class="form-label fw-bold small">Test Dataset (.csv)</label>
                        <div class="input-group">
                            <input type="text" class="form-control form-control-sm bg-light" readonly>
                            <button class="btn btn-outline-secondary btn-sm" type="button">Choose a file</button>
                        </div>
                    </div>
                    
                    <div class="col-12">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="confirmTrain">
                            <label class="form-check-label small" for="confirmTrain">I confirm that both training and test datasets are properly formatted</label>
                        </div>
                    </div>

                    <div class="col-12">
                        <div class="bg-light p-3 rounded d-flex align-items-center gap-3">
                            <button class="btn btn-secondary px-5 disabled" type="button">Start</button>
                            <div class="flex-grow-1">
                                <div class="progress" style="height: 8px;">
                                    <div class="progress-bar bg-success" style="width: 65%;"></div>
                                </div>
                            </div>
                            <span class="small fw-bold">Epoch 650/1000</span>
                        </div>
                    </div>
                </form>
            </div>
        </div>
    </div>
</div>