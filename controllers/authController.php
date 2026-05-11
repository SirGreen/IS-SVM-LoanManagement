<?php 

class AuthController {

    public function __construct() {
    }

    public function login() {
        if ($_SERVER['REQUEST_METHOD'] === 'POST') {
            $email = filter_var($_POST['email'], FILTER_SANITIZE_EMAIL);
            $password = $_POST['password'];

            if ($email==='user@example.com'){
                $_SESSION['user_id'] = 3;
                $_SESSION['role'] = 'user';
                header("Location: index.php?page=user");
            } elseif ($email === 'employer@example.com') {
                $_SESSION['user_id'] = 2;
                $_SESSION['role'] = 'employer';
                header("Location: index.php?page=employer");
            } else {
                $_SESSION['user_id'] = 1;
                $_SESSION['role'] = 'data_scientist';
                header("Location: index.php?page=model_workshop");
            }
        }
    }

    public function logout() {
        // Clear all session variables
        $_SESSION = array();

        // Destroy the session cookie
        if (ini_get("session.use_cookies")) {
            $params = session_get_cookie_params();
            setcookie(session_name(), '', time() - 42000,
                $params["path"], $params["domain"],
                $params["secure"], $params["httponly"]
            );
        }

        // Destroy the session
        session_destroy();

        // Redirect to login page
        header("Location: index.php?page=login");
        exit();
    }
}
?>