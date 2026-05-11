<?php
class WorkshopController {
    private $api;

    public function __construct() {
        $this->api = new APIService();
    }

    public function handle_request($action) {
        include __DIR__ . '/../views/header.php';
        switch ($action) {
            case 'dashboard':
                include '../views/data_scientist/model_workshop.php';
                break;

            case 'apply':
                break;
                
            case 'details':
                break;

            case 'download':
                break;
            
            default:
                break;
        }
    }
    

    public function train() {
        if ($_SERVER['REQUEST_METHOD'] === 'POST') {
            $this->api->call('model/train', 'POST', ['timestamp' => date('Y-m-d H:i:s')]);
            header('Location: index.php?page=model&status=training_started');
        }
    }
}