<?php
session_start();

include '../views/head.php';

// Auth
if (!isset($_SESSION['user_id']) && $_GET['page'] !== 'login') {
    header('Location: index.php?page=login');
    exit;
}

require_once '../models/apiService.php';

$page = $_GET['page'] ?? 'dashboard';
$action = $_GET['action'] ?? 'index';
$id = $_GET['id'] ?? null;

// Routing
switch ($page) {
    case 'employer':
        require_once '../controllers/employerController.php';
        $controller = new EmployerController();
        $action = $_GET['action'] ?? 'dashboard';
        $controller->handle_request($action);
        break;
    case 'user':
        require_once '../controllers/userController.php';
        $controller = new UserController();
        $action = $_GET['action'] ?? 'dashboard';
        $controller->handle_request($action);
        break;
    case 'model_workshop':
        require_once '../controllers/scientistController.php';
        $controller = new WorkshopController();
        $action = $_GET['action'] ?? 'dashboard';
        $controller->handle_request($action);
        break;
    case 'run_model':
        require_once '../controllers/runModel.php';
        $controller = new RunModel();
        $action = $_GET['action'] ?? '';
        $controller->handle_request($action);
        break;
    case 'login':
        if (isset($_POST['email'])) {
            require_once __DIR__ . '/../controllers/authController.php';
            $auth = new AuthController();
            $auth->login();
            exit();
        } else {
            include '../views/login.php';
        }
        break;
    case 'logout':
        require_once __DIR__ . '/../controllers/authController.php';
        $auth = new AuthController();
        $auth->logout();
        break;
    default:
        echo "404 - Page not found";
        break;
}
include __DIR__. '/../views/footer.php'
?>