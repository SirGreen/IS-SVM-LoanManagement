<?php
class RunModel {
    private $api;

    public function __construct() {
        $this->api = new APIService();
    }
    
    public function handle_request($action) {
        include __DIR__ . '/../views/header.php';
        $endpoint = "api/v1/predict";
        $data = $_POST;
        $_SESSION['form_data'] = $_POST;
        $jsonData = json_encode($data);
        $response = $this->api->call($endpoint,'POST',$jsonData);
        $_SESSION['api_results'] = $response;
        switch ($action) {
            case 'official':
                /* Store in db */
                break;
        
            default:
                break;
        }
        $user_role = $_SESSION['role'];
        header('Location: index.php?page='.$user_role.'&action=show_results');
    }
}