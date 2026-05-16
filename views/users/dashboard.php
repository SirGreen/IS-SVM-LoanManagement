<?php
function get_status_class($status) {
    return match($status) {
        'Pending'  => 'bg-warning text-dark',
        'Approved' => 'bg-success text-white',
        'Refused'  => 'bg-danger text-white',
        default    => 'bg-secondary'
    };
}
?>

<div class="container py-5" style="background-color: #f4f4f4; border-radius: 15px;">
    
    <div class="row justify-content-center mb-4">
        <div class="col-md-8">
            <div class="card shadow-sm border-0 text-center p-3">
                <div class="d-flex justify-content-between align-items-center px-4">
                    <div class="flex-grow-1">
                        <h4 class="fw-bold mb-0">Current Loan</h4>
                        <h2 class="text-primary fw-bold"><?= $current_loan ?></h2>
                    </div>
                    <a href="index.php?page=user&action=loan_details" class="btn btn-secondary btn-sm rounded-pill px-4">See details</a>
                </div>
            </div>
        </div>
    </div>

    <div class="row g-4 mb-5">
        <div class="col-md-6">
            <div class="card h-100 shadow-sm border-0 p-4 text-center">
                <h3 class="fw-bold">Make a simulation</h3>
                <p class="text-muted">Check your eligibility without impacting your score. Get an immediate answer from our AI.</p>
                <div class="mt-auto">
                    <a href="index.php?page=user&action=simulation" class="btn btn-primary rounded-pill px-5 py-2 w-75 fw-bold">Start Simulation</a>
                </div>
            </div>
        </div>
        <div class="col-md-6">
            <div class="card h-100 shadow-sm border-0 p-4 text-center">
                <h3 class="fw-bold">Apply for a Loan</h3>
                <p class="text-muted">Ready to make your project a reality? Fill out the complete form for an official review.</p>
                <div class="mt-auto">
                    <a href="index.php?page=user&action=apply" class="btn btn-outline-primary rounded-pill px-5 py-2 w-75 fw-bold">Apply</a>
                </div>
            </div>
        </div>
    </div>

    <div class="card shadow-sm border-0 p-4">
        <h3 class="fw-bold mb-4">Request history</h3>
        <div class="table-responsive">
            <table class="table align-middle">
                <thead class="text-muted">
                    <tr>
                        <th>Application ID</th>
                        <th>Type</th>
                        <th>Amount</th>
                        <th>Date</th>
                        <th>State</th>
                        <th>Results</th>
                    </tr>
                </thead>
                <tbody>
                    <?php foreach ($request_history as $app): ?>
                    <tr>
                        <td><?= $app['application_id'] ?></td>
                        <td><?= $app['type'] ?></td>
                        <td class="fw-bold"><?= $app['amount'] ?>$</td>
                        <td><?= $app['date'] ?></td>
                        <td>
                            <span class="badge rounded-pill px-3 <?= get_status_class($app['status']) ?>">
                                <?= $app['status'] ?>
                            </span>
                        </td>
                        <td>
                            <?php if ($app['status'] !== "Pending"): ?>
                                <?php if ($app['type'] === "Simulation"): ?>
                                    <a href="index.php?page=user&action=details&id=<?= urlencode($app['application_id'])?>" class="text-primary fw-bold text-decoration-none">Details</a>
                                <?php else: ?>
                                    <a href="#" class="text-primary fw-bold text-decoration-none">Download</a>
                                <?php endif; ?>
                            <?php else: ?>
                                -
                            <?php endif; ?>
                        </td>
                    </tr>
                    <?php endforeach; ?>
                </tbody>
            </table>
        </div>
    </div>
</div>