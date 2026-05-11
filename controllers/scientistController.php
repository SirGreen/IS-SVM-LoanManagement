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
                $stats = $this->get_stats();
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
            $endpoint = "api/v1/admin/retrain";
            $api_key = "api_key";
            $data = $_POST;
            $json_data = json_encode($data);
            $response = $this->api->call($endpoint,'POST',$json_data,$api_key);
            return $response;          
        }
    }

    private function get_margin(){
        $endpoint = "api/v1/admin/uncertainty-margin";
        $api_key = "api_key";
        $response = $this->api->call($endpoint,'GET',null,$api_key);
        return $response;
    }

    private function set_margin(){
        $endpoint = "api/v1/admin/uncertainty-margin";
        $api_key = "api_key";
        $data = $_POST;
        $json_data = json_encode($data);
        $response = $this->api->call($endpoint,'POST',$json_data,$api_key);
        return $response;
    }

    private function get_stats(){
        $endpoint = "api/v1/admin/stats";
        $api_key = "api_key";
        $response = $this->api->call($endpoint,'GET',null,$api_key);
        return $response;
    }
}