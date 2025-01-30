package com.example.testtry

import android.hardware.*
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.clickable
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.Alignment
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import androidx.navigation.compose.*
import kotlinx.coroutines.*
import org.eclipse.paho.client.mqttv3.*
import org.json.JSONArray
import org.json.JSONObject
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import android.widget.Toast
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.graphics.Color


class MainActivity : ComponentActivity(), SensorEventListener {

    private lateinit var mqttClient: MqttClient
    private lateinit var mqttOptions: MqttConnectOptions
    private lateinit var sensorManager: SensorManager
    private var lastSentTime = System.currentTimeMillis()

    private val batchInterval = 100L // Send data every 100 ms

    private var clientNameState by mutableStateOf("Not Connected")
    private var connectionStatus by mutableStateOf("Not Connected")
    private val availableSensors = mutableStateListOf<SensorData>()

    private val coroutineScope = CoroutineScope(Dispatchers.Main)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            val navController = rememberNavController()

            NavHost(navController, startDestination = "mqtt_demo") {
                composable("mqtt_demo") {
                    MqttDemoScreen(
                        clientName = clientNameState,
                        connectionStatus = connectionStatus,
                        availableSensors = availableSensors,
                        onConnectButtonClick = { brokerUrl, clientId, topic ->
                            if (connectionStatus == "Not Connected" || connectionStatus == "Disconnected") {
                                coroutineScope.launch {
                                    initializeMqttClient(brokerUrl, clientId, topic)
                                }
                            } else {
                                coroutineScope.launch {
                                    disconnectMqttClient()
                                }
                            }
                        },
                        onShowSensorsButtonClick = {
                            // Sensors are already initialized, no need to call initializeSensors here
                            navController.navigate("sensor_list")
                        },
                        navController = navController
                    )
                }
                composable("sensor_list") {
                    SensorListScreen(
                        availableSensors = availableSensors,
                        navController = navController
                    )
                }
                composable("sensor_details/{sensorName}") { backStackEntry ->
                    val sensorName = backStackEntry.arguments?.getString("sensorName")
                    val sensor = availableSensors.find { it.name == sensorName }
                    if (sensor != null) {
                        SensorDetailsScreen(sensor = sensor, navController = navController)
                    }
                }
            }
        }

        // Initialize sensors as soon as the activity is created
        coroutineScope.launch {
            initializeSensors()
        }
    }

    private suspend fun initializeMqttClient(brokerUrl: String, clientId: String, topic: String) {
        var reconnectAttempts = 0
        val maxReconnectAttempts = 5
        var connectionSuccess = false

        // Show reconnecting message
        withContext(Dispatchers.Main) {
            connectionStatus = "Connecting..."
        }

        while (reconnectAttempts < maxReconnectAttempts && !connectionSuccess) {
            try {
                mqttClient = MqttClient(brokerUrl, clientId, null)
                mqttOptions = MqttConnectOptions().apply {
                    isCleanSession = true
                    connectionTimeout = 10
                }

                mqttClient.connect(mqttOptions)
                connectionSuccess = true

                withContext(Dispatchers.Main) {
                    connectionStatus = "Connected"
                }

                mqttClient.setCallback(object : MqttCallback {
                    override fun messageArrived(topic: String?, message: MqttMessage?) {
                        message?.let {
                            val msg = it.toString()
                            Log.d("MQTT", "Received message: $msg")
                        }
                    }

                    override fun connectionLost(cause: Throwable?) {
                        Log.d("MQTT", "MQTT connection lost: ${cause?.message}")
                        CoroutineScope(Dispatchers.Main).launch {
                            connectionStatus = "Disconnected"
                        }
                    }

                    override fun deliveryComplete(token: IMqttDeliveryToken?) {
                        Log.d("MQTT", "Message delivered successfully")
                    }
                })

                mqttClient.subscribe(topic)

            } catch (e: MqttException) {
                reconnectAttempts++
                Log.e("MQTT", "Connection attempt $reconnectAttempts failed: ${e.message}")
                withContext(Dispatchers.Main) {
                    connectionStatus = "Reconnecting... ($reconnectAttempts/$maxReconnectAttempts)"
                }
                delay(3000) // Retry after 3 seconds
            }
        }

        if (!connectionSuccess) {
            withContext(Dispatchers.Main) {
                connectionStatus = "Failed to Connect"
            }
        }
    }


    private suspend fun initializeSensors() {
        withContext(Dispatchers.IO) {
            sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
            val allSensors = sensorManager.getSensorList(Sensor.TYPE_ALL)

            // Clear the previous sensors if any
            availableSensors.clear()

            // Define the sensor types you want to filter
            val requiredSensorTypes = listOf(
                Sensor.TYPE_ACCELEROMETER,
                Sensor.TYPE_GYROSCOPE,
                Sensor.TYPE_MAGNETIC_FIELD,
                Sensor.TYPE_AMBIENT_TEMPERATURE,
                Sensor.TYPE_PROXIMITY,
                Sensor.TYPE_PRESSURE,
                Sensor.TYPE_RELATIVE_HUMIDITY,
                Sensor.TYPE_LIGHT,
                Sensor.TYPE_MAGNETIC_FIELD_UNCALIBRATED
            )

            // Add only the required sensors to the list
            for (sensor in allSensors) {
                if (sensor.type in requiredSensorTypes) {
                    val sensorData = SensorData(sensor.name, sensor.type, emptyList())
                    availableSensors.add(sensorData)
                    try {
                        sensorManager.registerListener(this@MainActivity, sensor, SensorManager.SENSOR_DELAY_NORMAL)
                    } catch (e: Exception) {
                        Log.e("Sensor Init", "Error registering sensor: ${sensor.name}")
                    }
                }
            }
        }
    }

    private suspend fun disconnectMqttClient() {
        try {
            mqttClient.disconnect()
            withContext(Dispatchers.Main) {
                connectionStatus = "Disconnected"
            }
            Log.d("MQTT", "Disconnected from the broker")
        } catch (e: MqttException) {
            e.printStackTrace()
            withContext(Dispatchers.Main) {
                connectionStatus = "Failed to Disconnect"
            }
        }
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null) return

        val currentTime = System.currentTimeMillis()

        if (currentTime - lastSentTime < batchInterval) {
            return
        }

        lastSentTime = currentTime

        // Update only the specific sensor in availableSensors
        val updatedSensorList = availableSensors.map {
            if (it.name == event.sensor.name) {
                it.copy(values = event.values.toList()) // Update specific sensor value
            } else {
                it
            }
        }

        // Directly update the state
        availableSensors.clear()
        availableSensors.addAll(updatedSensorList)

        // Convert the updated sensor data to JSON
        val allSensorsJson = JSONObject().apply {
            val sensorsArray = JSONArray()
            for (sensor in availableSensors) {
                val sensorJson = JSONObject().apply {
                    put("sensor_name", sensor.name)
                    put("sensor_type", sensor.type)
                    put("values", JSONArray(sensor.values.map { it }))
                }
                sensorsArray.put(sensorJson)
            }
            put("sensors", sensorsArray)
            put("timestamp", currentTime)
        }

        // Log and send the updated sensor data to MQTT
        Log.d("Sensor Data", allSensorsJson.toString())
        sendSensorDataToMqtt(allSensorsJson)
    }



    private fun sendSensorDataToMqtt(allSensorsJson: JSONObject) {
        if (::mqttClient.isInitialized) {
            val message = MqttMessage(allSensorsJson.toString().toByteArray())
            CoroutineScope(Dispatchers.IO).launch {
                try {
                    mqttClient.publish("test/all_sensors", message)
                } catch (e: MqttException) {
                    e.printStackTrace()
                }
            }
        } else {
            Log.e("MQTT", "mqttClient is not initialized")
        }
    }


    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
    }


    override fun onDestroy() {
        super.onDestroy()
        coroutineScope.cancel()
        sensorManager.unregisterListener(this)

        CoroutineScope(Dispatchers.IO).launch {
            try {
                mqttClient.disconnect()
            } catch (e: MqttException) {
                e.printStackTrace()
            }
        }
    }
}

@Composable
fun MqttDemoScreen(
    clientName: String,
    connectionStatus: String,
    availableSensors: List<SensorData>,
    onConnectButtonClick: (String, String, String) -> Unit,
    onShowSensorsButtonClick: () -> Unit,
    navController: NavController
) {
    var brokerUrl by remember { mutableStateOf("tcp://10.0.2.2:1883") }
    var clientId by remember { mutableStateOf("AndroidClient") }
    var topic by remember { mutableStateOf("test/all_sensors") }

    val connectionStatusColor = when (connectionStatus) {
        "Connected" -> Color.Green
        "Disconnected" -> Color.Red
        else -> Color.Red
    }

    val context = LocalContext.current
    LaunchedEffect(connectionStatus) {
        when (connectionStatus) {
            "Connecting..." -> Toast.makeText(context, "Connecting...", Toast.LENGTH_SHORT).show()
            "Connected" -> Toast.makeText(context, "Connected to MQTT Broker", Toast.LENGTH_SHORT).show()
            "Disconnected" -> Toast.makeText(context, "Disconnected", Toast.LENGTH_SHORT).show()
        }
    }

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Text("SensorView", fontWeight = FontWeight.Bold, fontSize = 26.sp)
        Spacer(modifier = Modifier.height(16.dp))

        Row(modifier = Modifier.fillMaxWidth()) {
            Text("Connection Status: ", fontWeight = FontWeight.Bold, fontSize = 20.sp)
            Text(connectionStatus, color = connectionStatusColor, fontWeight = FontWeight.Bold, fontSize = 20.sp)
        }

        Spacer(modifier = Modifier.height(16.dp))

        OutlinedTextField(
            value = brokerUrl,
            onValueChange = { brokerUrl = it },
            label = { Text("Broker URL", fontWeight = FontWeight.Bold, fontSize = 18.sp) },
            modifier = Modifier.fillMaxWidth(),
            textStyle = TextStyle(fontSize = 18.sp)
        )

        Spacer(modifier = Modifier.height(8.dp))

        OutlinedTextField(
            value = clientId,
            onValueChange = { clientId = it },
            label = { Text("Client ID", fontWeight = FontWeight.Bold, fontSize = 18.sp) },
            modifier = Modifier.fillMaxWidth(),
            textStyle = TextStyle(fontSize = 18.sp)
        )

        Spacer(modifier = Modifier.height(8.dp))

        OutlinedTextField(
            value = topic,
            onValueChange = { topic = it },
            label = { Text("Topic", fontWeight = FontWeight.Bold, fontSize = 18.sp) },
            modifier = Modifier.fillMaxWidth(),
            textStyle = TextStyle(fontSize = 18.sp)
        )

        Spacer(modifier = Modifier.height(16.dp))

        Button(
            onClick = { onConnectButtonClick(brokerUrl, clientId, topic) },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(
                text = if (connectionStatus == "Connected") "Disconnect" else "Connect to MQTT",
                fontWeight = FontWeight.Bold,
                fontSize = 20.sp
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        Button(
            onClick = { onShowSensorsButtonClick() },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(
                text = "Show Available Sensors",
                fontWeight = FontWeight.Bold,
                fontSize = 20.sp
            )
        }
    }
}

@Composable
fun SensorListScreen(
    availableSensors: List<SensorData>,
    navController: NavController
) {
    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp)
    ) {
        // Back button to go back to the main screen
        Button(modifier = Modifier.fillMaxWidth(), onClick = { navController.popBackStack() }) {
            Text("Back to Main Screen", fontWeight = FontWeight.Bold, fontSize = 18.sp)
        }

        Spacer(modifier = Modifier.height(16.dp))

        if (availableSensors.isEmpty()) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Text("No sensors available", fontWeight = FontWeight.Bold, fontSize = 20.sp)
            }
        } else {
            LazyColumn(modifier = Modifier.fillMaxSize()) {
                items(availableSensors) { sensor ->
                    SensorItem(sensor = sensor, onClick = {
                        navController.navigate("sensor_details/${sensor.name}")
                    })
                }
            }
        }
    }
}

@Composable
fun SensorItem(sensor: SensorData, onClick: (SensorData) -> Unit) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp)
            .clickable { onClick(sensor) }
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Sensor Name: ${sensor.name}",
                fontWeight = FontWeight.Bold,
                fontSize = 20.sp
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "Sensor Type: ${sensor.type}",
                fontWeight = FontWeight.Bold,
                fontSize = 18.sp
            )
        }
    }
}

@Composable
fun SensorDetailsScreen(sensor: SensorData, navController: NavController) {
    // Keep track of the current sensor values in a mutable state
    var sensorValue by remember { mutableStateOf(sensor.values) }

    // Update the sensorValue when the parent list (availableSensors) is updated
    val availableSensors = remember { sensor.values }

    // Collect sensor value updates
    LaunchedEffect(sensor) {
        snapshotFlow { sensor.values }
            .collect { newValues ->
                sensorValue = newValues // Update the state when the values change
            }
    }

    Column(modifier = Modifier.padding(16.dp)) {
        // Back button to go back to the sensor list
        Button(
            modifier = Modifier.fillMaxWidth(),
            onClick = { navController.popBackStack() }
        ) {
            Text("Back to Available Sensors", fontWeight = FontWeight.Bold, fontSize = 18.sp)
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Sensor details inside a Card
        Card(modifier = Modifier.fillMaxWidth().padding(8.dp)) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Sensor Details", fontWeight = FontWeight.Bold, fontSize = 24.sp)
                Spacer(modifier = Modifier.height(16.dp))
                Text("Sensor Name: ${sensor.name}", fontWeight = FontWeight.Bold, fontSize = 20.sp)
                Text("Sensor Type: ${sensor.type}", fontWeight = FontWeight.Bold, fontSize = 20.sp)

                // Display the actual sensor values
                Text("Sensor Values: ${sensorValue.joinToString(", ")}", fontWeight = FontWeight.Bold, fontSize = 20.sp)
            }
        }
    }
}

data class SensorData(
    val name: String,
    val type: Int,
    var values: List<Float> = emptyList()
)
