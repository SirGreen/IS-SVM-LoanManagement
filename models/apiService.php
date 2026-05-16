<?php
class APIService {
    private $baseUrl = "http://localhost:8000"; // URL API FastAPI

    public function call($endpoint, $method = 'GET', $data = null, $api_key = null) {
        $url = $this->baseUrl . '/' . ltrim($endpoint, '/');
    
        // GET
        if ($method === 'GET' && !empty($data)) {
            $url .= '?' . http_build_query($data);
        }

        $ch = curl_init($url);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_CUSTOMREQUEST, $method);

        // POST
        if ($method !== 'GET' && $data !== null) {
            $payload = is_array($data) ? json_encode($data) : $data;
            curl_setopt($ch, CURLOPT_POSTFIELDS, $payload);
        }

        $headers = ['Content-Type: application/json'];
        if ($api_key !== null) {
            $headers[] = "X-API-Key: " . $api_key;
        }

        curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);

        if (curl_errno($ch)) {
            echo 'Error cURL : ' . curl_error($ch);
        }
        curl_close($ch);

        return [
            'status' => $httpCode,
            'data' => json_decode($response, true)
        ];
    }
}
?>