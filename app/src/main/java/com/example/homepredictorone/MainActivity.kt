package com.example.homepredictorone

import android.os.Bundle
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
// If you change R.id.outputnum1 to a TextView in your XML:
// import android.widget.TextView
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import java.nio.FloatBuffer

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main) // Ensure this layout file exists
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        // --- UI Element Initialization ---
        // Ensure these IDs match your activity_main.xml
        val inputArea = findViewById<EditText>(R.id.inputnum1)
        val inputRooms = findViewById<EditText>(R.id.inputnum2)
        val inputZone = findViewById<EditText>(R.id.inputnum3)
        val pressToPredict = findViewById<Button>(R.id.predict_button)
        val resultText = findViewById<EditText>(R.id.outputnum1) // Or TextView
        // Example if using TextView for result:
        // val resultText = findViewById<TextView>(R.id.outputnum1)

        // --- Prediction Button Logic ---
        pressToPredict.setOnClickListener {
            // 1. Get input data and convert to Float
            val areaData = inputArea.text.toString().toFloatOrNull()
            val roomData = inputRooms.text.toString().toFloatOrNull()
            val zoneData = inputZone.text.toString().toFloatOrNull()

            // 2. Validate inputs
            if (areaData != null && roomData != null && zoneData != null) {
                try {
                    // 3. Initialize ONNX Runtime environment and session
                    val ortEnvironment = OrtEnvironment.getEnvironment()
                    val ortSession = createORTSession(ortEnvironment)

                    if (ortSession == null) {
                        resultText.setText("Error: Could not create ONNX session.")
                        Toast.makeText(this, "Failed to initialize the prediction model.", Toast.LENGTH_LONG).show()
                        return@setOnClickListener // Exit if session creation failed
                    }

                    // 4. Execute the model with the 3 input variables
                    val output = executeModel(areaData, roomData, zoneData, ortSession, ortEnvironment)

                    // 5. Display the result or error
                    if (output != null) {
                        resultText.setText("Predicted Price is $${String.format("%.2f", output)}") // Format to 2 decimal places
                        // For TextView: resultText.text = "Predicted Price is $${String.format("%.2f", output)}"
                    } else {
                        resultText.setText("Error in prediction")
                        Toast.makeText(this, "Could not get prediction. Model might have returned null or an error occurred.", Toast.LENGTH_SHORT).show()
                    }
                } catch (e: Exception) {
                    resultText.setText("Error during prediction process")
                    Toast.makeText(this, "An unexpected error occurred: ${e.localizedMessage}", Toast.LENGTH_LONG).show()
                    e.printStackTrace() // Log the full error for debugging
                }
            } else {
                Toast.makeText(this, "Invalid Input. Please enter valid numbers for all three fields.", Toast.LENGTH_LONG).show()
            }
        }
    }

    /**
     * Creates an ONNX Runtime session from the model file in res/raw.
     */
    private fun createORTSession(ortEnvironment: OrtEnvironment?): OrtSession? {
        val currentOrtEnvironment = ortEnvironment ?: OrtEnvironment.getEnvironment() ?: run {
            System.err.println("DEBUG: OrtEnvironment is null, cannot create session.")
            return null
        }
        return try {
            // Ensure you have 'house_price_model.onnx' in your 'res/raw' folder
            resources.openRawResource(R.raw.house_price_model).use { modelInputStream ->
                val modelBytes = modelInputStream.readBytes()
                currentOrtEnvironment.createSession(modelBytes)
            }
        } catch (e: Exception) {
            System.err.println("DEBUG: Error loading ONNX model or creating session.")
            e.printStackTrace()
            Toast.makeText(this, "Error loading model or creating session: ${e.localizedMessage}", Toast.LENGTH_LONG).show()
            null
        }
    }

    /**
     * Executes the ONNX model with three input features.
     */
    private fun executeModel(
        area: Float,
        rooms: Float,
        zone: Float,
        ortSession: OrtSession,
        ortEnvironment: OrtEnvironment
    ): Float? {
        System.err.println("DEBUG: executeModel called with area=$area, rooms=$rooms, zone=$zone")

        val inputName = ortSession.inputNames?.firstOrNull() ?: run {
            System.err.println("DEBUG: Input name not found in ONNX model")
            return null
        }
        System.err.println("DEBUG: Using input name: $inputName")

        val inputArray = floatArrayOf(area, rooms, zone)
        val floatBufferInput = FloatBuffer.wrap(inputArray)
        val tensorShape = longArrayOf(1, 3)
        val tensorInput = OnnxTensor.createTensor(ortEnvironment, floatBufferInput, tensorShape)

        val outputValue: Float?
        try {
            System.err.println("DEBUG: About to run model inference with input: ${inputArray.joinToString()}")
            ortSession.run(mapOf(inputName to tensorInput)).use { result ->
                System.err.println("DEBUG: Model inference complete. Result object: $result")

                val firstResultOnnxValue = result.firstOrNull() // This is an OnnxValue
                System.err.println("DEBUG: First result entry (OnnxValue): $firstResultOnnxValue")

                // The 'value' of the OnnxValue can be an OnnxTensor, OnnxSequence, or OnnxMap
                // In your case, Logcat shows it's an OnnxTensor.
                val outputOnnxTensor = firstResultOnnxValue?.value as? ai.onnxruntime.OnnxTensor
                System.err.println("DEBUG: Output OnnxTensor from result: $outputOnnxTensor")
                System.err.println("DEBUG: Type of output OnnxTensor: ${outputOnnxTensor?.javaClass?.canonicalName}")


                if (outputOnnxTensor != null) {
                    // Now, get the actual data from the OnnxTensor.
                    // Its 'value' property will be the multi-dimensional array.
                    // Given shape [1,1] and javaType FLOAT, its value will be Array<FloatArray>
                    val nestedArrayValue = outputOnnxTensor.value as? Array<FloatArray>
                    System.err.println("DEBUG: Nested array value from OnnxTensor: $nestedArrayValue")

                    if (nestedArrayValue != null && nestedArrayValue.isNotEmpty() && nestedArrayValue[0].isNotEmpty()) {
                        val predicted = nestedArrayValue[0][0]
                        System.err.println("DEBUG: Successfully extracted output: $predicted")
                        outputValue = predicted
                    } else {
                        System.err.println("DEBUG: Failed to extract value from nestedArrayValue. nestedArrayValue was: $nestedArrayValue")
                        outputValue = null
                    }
                } else {
                    System.err.println("DEBUG: Could not get OnnxTensor from result.")
                    outputValue = null
                }
            }
        } catch (e: Exception) {
            System.err.println("DEBUG: Exception during model execution or result processing.")
            e.printStackTrace()
            Toast.makeText(this, "Error during model execution: ${e.localizedMessage}", Toast.LENGTH_LONG).show()
            return null
        } finally {
            tensorInput.close()
        }

        System.err.println("DEBUG: Returning outputValue from executeModel: $outputValue")
        return outputValue
    }
}
