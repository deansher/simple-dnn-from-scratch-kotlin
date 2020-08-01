/*
 * This Kotlin source file was generated by the Gradle 'init' task.
 */
package parts.wisdom.simplednn

import koma.extensions.*
import koma.internal.default.generated.ndarray.DefaultGenericNDArray
import koma.matrix.Matrix
import koma.ndarray.NDArray
import koma.rand
import koma.zeros
import kotlin.math.max
import kotlin.math.pow

private const val LEARNING_RATE = 3e-3
private const val MAX_INITIAL_VALUE = 1.0

abstract class HiddenLayer(
    val inputShape: Shape,
    val outputShape: Shape
) {
    /**
     * Compute this `Layer`'s output from its input.
     */
    abstract operator fun invoke(bottomInput: Matrix<Double>): Matrix<Double>

    /**
     * Make a stateful trainer that will process one batch of inputs.
     */
    abstract fun makeBatchTrainer(): HiddenLayerBatchTrainer
}

interface HiddenLayerBatchTrainer {
    // TODO: Refactor `train` to free up that name and use it for this function instead.
    fun train(
        bottomInput: Matrix<Double>,
        dLossDOutput: Matrix<Double>
    )

    fun updateParameters()
}

abstract class OutputLayer {
    /**
     * Compute this `Layer`'s output from its input.
     */
    abstract operator fun invoke(input: Matrix<Double>): Matrix<Double>

    /**
     * Make a stateful trainer that will process one batch of inputs.
     */
    abstract fun makeBatchTrainer(): OutputLayerBatchTrainer
}

interface OutputLayerBatchTrainer {
    fun train(
        bottomInput: Matrix<Double>,
        label: Coords
    )

    fun updateParameters()
}

class InputLayer(shape: Shape) : HiddenLayer(shape, shape) {
    override fun invoke(bottomInput: Matrix<Double>): Matrix<Double> = bottomInput

    override fun makeBatchTrainer(): HiddenLayerBatchTrainer {
        return object : HiddenLayerBatchTrainer {
            override fun train(bottomInput: Matrix<Double>, dLossDOutput: Matrix<Double>) {
            }

            override fun updateParameters() {
            }
        }
    }

}

/**
 * A linear fully connected layer (no activation function).
 */
class FullyConnected(
    val source: HiddenLayer,
    outputShape: Shape
) : HiddenLayer(source.outputShape, outputShape) {
    /**
     * For each output, a weight for each input.
     */
    private var weight: NDArray<Matrix<Double>> =
        makeArrayOfMatrices(outputShape) { _, _ ->
            rand(inputShape.numRows, inputShape.numCols) * MAX_INITIAL_VALUE
        }

    /**
     * A bias for each output.
     */
    private var bias: Matrix<Double> = rand(outputShape) * MAX_INITIAL_VALUE

    override operator fun invoke(bottomInput: Matrix<Double>): Matrix<Double> {
        val myInput = source(bottomInput)
        val weightedSums =
            Matrix(
                outputShape.numRows,
                outputShape.numCols
            ) { row, col ->
                weight[row, col] dot myInput
            }
        return weightedSums + bias
    }

    override fun makeBatchTrainer(): HiddenLayerBatchTrainer =
        object : HiddenLayerBatchTrainer {
            private var sourceTrainer = source.makeBatchTrainer()

            /**
             * For each input, a weight for each output.
             */
            private var outputWeight: NDArray<Matrix<Double>> =
                makeArrayOfMatrices(source.outputShape) { inputRow, inputCol ->
                    Matrix(
                        outputShape.numRows,
                        outputShape.numCols
                    ) { outputRow, outputCol ->
                        weight[outputRow, outputCol][inputRow, inputCol]
                    }
                }

            private val batchDeltaBias =
                zeros(
                    outputShape.numRows,
                    outputShape.numCols
                )
            private val batchDeltaWeight: NDArray<Matrix<Double>> =
                makeArrayOfMatrices(outputShape) { _, _ ->
                    zeros(inputShape.numRows, inputShape.numCols)
                }

            private var training = true

            override fun train(bottomInput: Matrix<Double>, dLossDOutput: Matrix<Double>) {
                check(training)
                val myInput = source(bottomInput)
                dLossDOutput.forEachIndexedN { id, d ->
                    val deltaBias = -LEARNING_RATE * d
                    batchDeltaBias[id[0], id[1]] += deltaBias

                    val dw = batchDeltaWeight[id[0], id[1]]
                    for (row in 0 until inputShape.numRows) {
                        for (col in 0 until inputShape.numCols) {
                            val dLossDW = d * myInput[row, col]
                            val deltaW = -LEARNING_RATE * dLossDW
                            dw[row, col] += deltaW
                        }
                    }
                }
                if (source !is InputLayer) {
                    val dLossDInput = Matrix(
                        source.outputShape.numRows,
                        source.outputShape.numCols
                    ) { inputRow, inputCol ->
                        dLossDOutput dot outputWeight[inputRow, inputCol]
                    }
                    sourceTrainer.train(bottomInput, dLossDInput)
                }
            }

            override fun updateParameters() {
                bias += batchDeltaBias
                weight.forEachIndexedN { iw, _ ->
                    weight[iw[0], iw[1]] += batchDeltaWeight[iw[0], iw[1]]
                }
                training = false
            }
        }
}

class Relu(val source: HiddenLayer, outputShape: Shape) : HiddenLayer(source.outputShape, outputShape) {
    override fun invoke(bottomInput: Matrix<Double>): Matrix<Double> {
        val myInput = source(bottomInput)
        return Matrix(
            outputShape.numRows,
            outputShape.numCols
        ) { row, col ->
            max(0.0, myInput[row, col])
        }
    }

    override fun makeBatchTrainer(): HiddenLayerBatchTrainer =
        object : HiddenLayerBatchTrainer {
            private var sourceTrainer = source.makeBatchTrainer()

            override fun train(bottomInput: Matrix<Double>, dLossDOutput: Matrix<Double>) {
                val myInput = source(bottomInput)
                val dLossDInput = Matrix(
                    outputShape.numRows,
                    outputShape.numCols
                ) { row, col ->
                    if (myInput[row, col] > 0) {
                        dLossDOutput[row, col]
                    } else {
                        0.0
                    }
                }
                sourceTrainer.train(bottomInput, dLossDInput)
            }

            override fun updateParameters() {
            }

        }
}

/**
 * Softmax classifier with cross-entropy loss.
 */
class Softmax(
    val source: HiddenLayer
) : OutputLayer() {

    fun inferClass(x: Example): Coords {
        val logits = source.invoke(x.matrix)

        var bestClass = Coords(0, 0)
        var bestLogit = logits[bestClass.row, bestClass.col]

        logits.forEachIndexedN { idx, logit ->
            if (logit > bestLogit) {
                bestClass = Coords(idx)
                bestLogit = logit
            }
        }
        return bestClass
    }

    override operator fun invoke(input: Matrix<Double>): Matrix<Double> {
        val logits = source(input)
        val es = logits.map { Math.E.pow(it) }
        val sumEs = es.elementSum()
        return es.map { it / sumEs }
    }

    override fun makeBatchTrainer(): OutputLayerBatchTrainer =
        object : OutputLayerBatchTrainer {
            private var sourceTrainer = source.makeBatchTrainer()
            private var training = true

            override fun train(
                bottomInput: Matrix<Double>,
                label: Coords
            ) {
                check(training)
                val ps = invoke(bottomInput)
                val dLossDLogit = Matrix(ps.numRows(), ps.numCols()) { row, col ->
                    val p = ps[row, col]
                    if (label.row == row && label.col == col) {
                        p - 1.0
                    } else {
                        p
                    }
                }
                sourceTrainer.train(bottomInput, dLossDLogit)
            }

            override fun updateParameters() {
                sourceTrainer.updateParameters()
                training = false
            }
        }
}

private fun rand(shape: Shape): Matrix<Double> = rand(shape.numRows, shape.numCols)

private infix fun Matrix<Double>.dot(m: Matrix<Double>) = (this emul m).elementSum()

fun makeArrayOfMatrices(
    shape: Shape,
    fill: (row: Int, col: Int) -> Matrix<Double>
): NDArray<Matrix<Double>> {
    return DefaultGenericNDArray(shape.numRows, shape.numCols) { coords ->
        fill(coords[0], coords[1])
    }
}



