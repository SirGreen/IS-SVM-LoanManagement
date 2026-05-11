<div class="d-flex justify-content-between align-items-center mb-4">
    <h2><i class="bi bi-briefcase me-2"></i>Espace Employé</h2>
    <div class="input-group w-25">
        <input type="text" class="form-control" placeholder="Rechercher une tâche...">
        <button class="btn btn-outline-secondary"><i class="bi bi-search"></i></button>
    </div>
</div>

<div class="card border-0 shadow-sm">
    <div class="table-responsive">
        <table class="table table-hover align-middle mb-0">
            <thead class="bg-light">
                <tr>
                    <th>ID</th>
                    <th>Tâche</th>
                    <th>Priorité</th>
                    <th>Status</th>
                    <th>Action</th>
                </tr>
            </thead>
            <tbody>
                <?php foreach ($tasks as $task): ?>
                <tr>
                    <td>#<?= $task['id'] ?></td>
                    <td><strong><?= $task['title'] ?></strong></td>
                    <td>
                        <span class="badge rounded-pill <?= $task['priority'] == 'High' ? 'bg-danger' : 'bg-info' ?>">
                            <?= $task['priority'] ?>
                        </span>
                    </td>
                    <td>
                        <select class="form-select form-select-sm w-auto">
                            <option <?= $task['status'] == 'En cours' ? 'selected' : '' ?>>En cours</option>
                            <option <?= $task['status'] == 'Terminé' ? 'selected' : '' ?>>Terminé</option>
                        </select>
                    </td>
                    <td>
                        <a href="index.php?page=tasks&action=details&id=<?= $task['id'] ?>" class="btn btn-sm btn-light">
                            <i class="bi bi-eye"></i>
                        </a>
                    </td>
                </tr>
                <?php endforeach; ?>
            </tbody>
        </table>
    </div>
</div>