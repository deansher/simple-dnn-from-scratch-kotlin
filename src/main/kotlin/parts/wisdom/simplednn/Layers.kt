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
import kotlin.math.pow

private const val LEARNING_RATE = 3e-3
private const val MAX_INITIAL_VALUE = 1.0

abstract class Layer(
    val inputShape: Shape,
    val outputShape: Shape
) {
    abstract fun makeTopLayerBatchTrainer(): TopLayerBatchTrainer
    abstract fun makeLowerLayerBatchTrainer(): LowerLayerBatchTrainer
    abstract fun computeOutput(input: Matrix<Double>): Matrix<Double>
}

interface TopLayerBatchTrainer {
    fun train(
        input: Matrix<Double>,
        label: Coords
    )
    fun updateParameters()
}

interface LowerLayerBatchTrainer {
    fun train(dLossDOutput: Matrix<Double>)
    fun updateParameters()
}

/**
 * Fully connected softmax classifier with cross-entropy loss.
 */
class FullyConnectedSoftmax(
    inputShape: Shape,
    outputShape: Shape
) : Layer(inputShape, outputShape) {
    private var bias: Matrix<Double> = rand(outputShape) * MAX_INITIAL_VALUE

    private var weight: NDArray<Matrix<Double>> =
        makeArrayOfMatrices(outputShape) { _, _ ->
            rand(inputShape.numRows, inputShape.numCols) * MAX_INITIAL_VALUE
        }

    fun inferClass(x: Example): Coords {
        val logits = computeLogits(x.matrix)

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

    private fun computeLogits(input: Matrix<Double>): Matrix<Double> {
        val weightedSums =
            Matrix(
                outputShape.numRows,
                outputShape.numCols
            ) { row, col ->
                weight[row, col] dot input
            }
        return weightedSums + bias
    }

    override fun computeOutput(input: Matrix<Double>): Matrix<Double> {
        val logits = computeLogits(input)
        val es = logits.map { Math.E.pow(it) }
        val sumEs = es.elementSum()
        return es.map { it / sumEs }
    }

    inner class MyTopLayerBatchTrainer: TopLayerBatchTrainer {
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

        override fun train(
            input: Matrix<Double>,
            label: Coords
        ) {
            check(training)
            val ps = computeOutput(input)
            ps.forEachIndexedN { ip, p ->
                val dLossDLogit =
                    if (label.equalsIdx(ip)) {
                        p - 1.0
                    } else {
                        p
                    }
                val deltaBias = -LEARNING_RATE * dLossDLogit
                batchDeltaBias[ip[0], ip[1]] += deltaBias

                val dw = batchDeltaWeight[ip[0], ip[1]]
                for (row in 0 until inputShape.numRows) {
                    for (col in 0 until inputShape.numCols) {
                        val dLossDW = dLossDLogit * input[row, col]
                        val deltaW = -LEARNING_RATE * dLossDW
                        dw[row, col] += deltaW
                    }
                }
            }
        }

        override fun updateParameters() {
            bias += batchDeltaBias
            weight.forEachIndexedN { idx, _ ->
                weight[idx[0], idx[1]] += batchDeltaWeight[idx[0], idx[1]]
            }
            training = false
        }
    }

    override fun makeTopLayerBatchTrainer(): TopLayerBatchTrainer =
        MyTopLayerBatchTrainer()

    override fun makeLowerLayerBatchTrainer(): LowerLayerBatchTrainer {
        TODO("Not yet implemented")
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



