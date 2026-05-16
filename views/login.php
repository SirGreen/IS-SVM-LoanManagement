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

<!-- ====== AUTH ====== -->
<main class="container mt-5">
    <div class="row justify-content-center">
        <div class="col-md-4">
            <h2 class="text-center">Welcome back</h2>
            <p class="text-center mb-4"> Sign in to your <strong>loanSVM</strong> account </p>
            <form method="POST" action="index.php?page=login">
                <div class="mb-3">
                    <label>Email</label>
                    <input type="email" name="email" class="form-control" required>
                </div>
                <div class="mb-3">
                    <label>Password</label>
                    <input type="password" name="password" class="form-control">
                </div>
                <button type="submit" class="btn btn-success w-100">Log In</button>
            </form>
        </div>
    </div>
    <!-- Demo Credentials -->
    <div class="container d-flex justify-content-center my-4">
        <div class="p-4 bg-success-subtle border border-success-subtle rounded text-center shadow-sm">
            <h6 class="text-success-emphasis fw-bold">Demo Accounts (no password required):</h6>
            <div class="text-secondary">
                <p class='fw-bold'>Admin:</p> scientist@example.com<br><br>
                <p class='fw-bold'>Employer:</p> employer@example.com<br><br>
                <p class='fw-bold'>Client:</p> user@example.com
            </div>
        </div>
    </div>
</main>