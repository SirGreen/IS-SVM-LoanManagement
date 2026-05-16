<div class="sidebar d-flex flex-column p-3">
    <h4 class="mb-4">LoanSVM</h4>
    <ul class="nav nav-pills flex-column mb-auto">
        <?php if (!isset($_SESSION['role'])) header("Location: index.php?page=login") ?>
        <?php if ($_SESSION['role']==='employer'):?>
            <li><a href="index.php?page=<?= $_SESSION['role']?>&action=dashboard" class="nav-link <?= $action=='dashboard'?'active':'' ?>">Dashboard</a></li>
            <li><a href="index.php?page=employer&action=loan" class="nav-link <?= $action=='loan'?'active':'' ?>">Make a Loan</a></li>
        <?php elseif ($_SESSION['role']==='data_scientist'):?>
            <li><a href="index.php?page=model_workshop" class="nav-link <?= $action=='workshop'?'active':'' ?>">Model Workshop</a></li>
        <?php else:?>
            <li><a href="index.php?page=<?= $_SESSION['role']?>&action=dashboard" class="nav-link <?= $action=='dashboard'?'active':'' ?>">Dashboard</a></li>
        <?php endif;?>
        <li><a href="index.php?page=logout" class="nav-link"> Logout</a></li>
    </ul>
</div>
<div class="main-content">
        