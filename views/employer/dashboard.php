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
        <div class="col-md-6">
            <div class="card border-0 shadow-sm p-4 h-100" style="border-left: 5px solid #ffc107 !important;">
                <div class="d-flex align-items-center">
                    <div class="rounded-circle bg-warning bg-opacity-10 p-3 me-3">
                        <span class="fs-3">⏳</span>
                    </div>
                    <div>
                        <h6 class="text-muted mb-1 text-uppercase small fw-bold">Pending Reviews</h6>
                        <h2 class="fw-bold mb-0"><?= $stats['pending_reviews'] ?></h2>
                    </div>
                </div>
            </div>
        </div>
        <div class="col-md-6">
            <div class="card border-0 shadow-sm p-4 h-100" style="border-left: 5px solid #198754 !important;">
                <div class="d-flex align-items-center">
                    <div class="rounded-circle bg-success bg-opacity-10 p-3 me-3">
                        <span class="fs-3">✅</span>
                    </div>
                    <div>
                        <h6 class="text-muted mb-1 text-uppercase small fw-bold">Completed This Month</h6>
                        <h2 class="fw-bold mb-0"><?= $stats['completed_this_month'] ?></h2>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="card border-0 shadow-sm p-4">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h4 class="fw-bold mb-0">Pending Reviews List</h4>
            <span class="badge bg-light text-dark border fw-normal"><?= count($pending_list) ?> files waiting</span>
        </div>
        
        <div class="table-responsive">
            <table class="table table-hover align-middle">
                <thead class="table-light">
                    <tr>
                        <th class="border-0">Application ID</th>
                        <th class="border-0">Amount</th>
                        <th class="border-0">Date Received</th>
                        <th class="border-0">Status</th>
                        <th class="border-0 text-end">Action</th>
                    </tr>
                </thead>
                <tbody>
                    <?php foreach ($pending_list as $review): ?>
                    <tr>
                        <td class="fw-bold text-primary"><?= $review['application_id'] ?></td>
                        <td class="fw-semibold"><?= $review['amount'] ?></td>
                        <td class="text-muted small"><?= $review['date'] ?></td>
                        <td>
                            <span class="badge rounded-pill bg-warning text-dark px-3 shadow-sm">
                                <?= $review['status'] ?>
                            </span>
                        </td>
                        <td class="text-end">
                            <button class="btn btn-outline-dark btn-sm rounded-pill px-3 fw-bold">Review File</button>
                        </td>
                    </tr>
                    <?php endforeach; ?>
                </tbody>
            </table>
        </div>
        
        <div class="d-flex justify-content-center mt-3">
            <nav>
                <ul class="pagination pagination-sm mb-0">
                    <li class="page-item disabled"><a class="page-link" href="#">Prev</a></li>
                    <li class="page-item active"><a class="page-link" href="#">1</a></li>
                    <li class="page-item"><a class="page-link" href="#">2</a></li>
                    <li class="page-item"><a class="page-link" href="#">Next</a></li>
                </ul>
            </nav>
        </div>
    </div>
</div>

<style>
    /* Amélioration du rendu des tables */
    .table thead th {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #6c757d;
    }
    .card {
        transition: transform 0.2s ease;
    }
    .card:hover {
        transform: translateY(-2px);
    }
</style>