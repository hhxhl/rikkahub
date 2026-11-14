package me.rerere.asr

/**
 * An ASR controller whose underlying service/recognizer survives multiple voice turns.
 *
 * [finishTurn] must stop an interrupted turn if needed, but must not destroy the underlying
 * recognizer. [dispose] still releases all resources when voice mode exits.
 */
interface ReusableASRController : ASRController {
    fun finishTurn()
}
