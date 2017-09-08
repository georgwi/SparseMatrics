package us.ihmc.sparseMatrices;

import gnu.trove.function.TDoubleFunction;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.procedure.TIntDoubleProcedure;

public class SparseMatrixInverter
{
   private static final double epsilon = 1.0e-10;

   private final SparseMatrix localMatrixStep1 = new SparseMatrix();
   private final SparseMatrix localMatrixStep2 = new SparseMatrix();
   private final LowestIndexFinder lowestIndexFinder = new LowestIndexFinder();

   private final SparseMatrix permutationMatrix = new SparseMatrix();
   private final SparseMatrix localInverse = new SparseMatrix();

   private final EliminationProcedure eliminationProcedure = new EliminationProcedure();
   private final RowScalingProcedure rowScalingProcedure = new RowScalingProcedure();

   public boolean invert(SparseMatrix matrix, SparseMatrix inverseToPack)
   {
      int n = matrix.getRows();
      if (n != matrix.getColumns())
      {
         throw new RuntimeException("Can only invert square matrices.");
      }

      localInverse.setIdentity(n);
      localMatrixStep1.set(matrix);

      permutationMatrix.resize(n, n);
      permutationMatrix.clear();

      // Step one: make matrix upper triangle.
      for (int i = 0; i < n; i++)
      {
         int pivotRowIndex = i;
         TIntDoubleHashMap pivotRow = localMatrixStep1.getRow(pivotRowIndex);

         if (pivotRow == null || pivotRow.isEmpty())
         {
            return false;
         }

         lowestIndexFinder.reset();
         pivotRow.forEachEntry(lowestIndexFinder);

         if (!lowestIndexFinder.valid())
         {
            return false;
         }

         int pivotColumnIndex = lowestIndexFinder.getLowestIndex();
         permutationMatrix.set(pivotColumnIndex, pivotRowIndex, 1.0);

         double pivot = pivotRow.get(pivotColumnIndex);
         if (closeToZero(pivot))
         {
            return false;
         }

         rowScalingProcedure.set(1.0 / pivot);
         pivotRow.transformValues(rowScalingProcedure);

         TIntDoubleHashMap pivotRowInverse = localInverse.getRow(pivotRowIndex);
         pivotRowInverse.transformValues(rowScalingProcedure);

         for (int lowerRowIdx = pivotRowIndex + 1; lowerRowIdx < n; lowerRowIdx++)
         {
            TIntDoubleHashMap lowerRow = localMatrixStep1.getRow(lowerRowIdx);
            if (lowerRow == null)
            {
               return false;
            }
            double eliminateValue = lowerRow.remove(pivotColumnIndex);
            if (closeToZero(eliminateValue))
            {
               continue;
            }

            eliminationProcedure.set(eliminateValue, pivotColumnIndex, lowerRow);
            pivotRow.forEachEntry(eliminationProcedure);

            TIntDoubleHashMap lowerRowInverse = localInverse.getRow(lowerRowIdx);
            eliminationProcedure.set(eliminateValue, -1, lowerRowInverse);
            pivotRowInverse.forEachEntry(eliminationProcedure);
         }
      }

      localMatrixStep2.mult(permutationMatrix, localMatrixStep1);
      inverseToPack.mult(permutationMatrix, localInverse);

      // Step two make original matrix match the identity.
      for (int i = 0; i < n; i++)
      {
         int modifiedRowIndex = i;
         TIntDoubleHashMap modifiedRow = localMatrixStep2.getRow(modifiedRowIndex);
         TIntDoubleHashMap modifiedRowInverse = inverseToPack.getRow(modifiedRowIndex);

         for (int lowerColumnIdx = modifiedRowIndex + 1; lowerColumnIdx < n; lowerColumnIdx++)
         {
            int modifiedColumnIndex = lowerColumnIdx;
            double eliminateValue = modifiedRow.remove(modifiedColumnIndex);
            if (closeToZero(eliminateValue))
            {
               continue;
            }

            TIntDoubleHashMap pivotRow = localMatrixStep2.getRow(modifiedColumnIndex);
            eliminationProcedure.set(eliminateValue, modifiedColumnIndex, modifiedRow);
            pivotRow.forEachEntry(eliminationProcedure);

            TIntDoubleHashMap pivotRowInverse = inverseToPack.getRow(modifiedColumnIndex);
            eliminationProcedure.set(eliminateValue, -1, modifiedRowInverse);
            pivotRowInverse.forEachEntry(eliminationProcedure);
         }
      }

      return true;
   }

   private static boolean closeToZero(double value)
   {
      return value < epsilon && value > -epsilon;
   }

   private class LowestIndexFinder implements TIntDoubleProcedure
   {
      private int lowestIndex;

      public void reset()
      {
         lowestIndex = Integer.MAX_VALUE;
      }

      public int getLowestIndex()
      {
         return lowestIndex;
      }

      public boolean valid()
      {
         return lowestIndex != Integer.MAX_VALUE;
      }

      @Override
      public boolean execute(int index, double value)
      {
         if (lowestIndex > index && !closeToZero(value))
         {
            lowestIndex = index;
         }
         return true;
      }
   }

   private class EliminationProcedure implements TIntDoubleProcedure
   {
      private double valueToElimiate;
      private int skipIndex;
      private TIntDoubleHashMap rowToModify;

      public void set(double valueToElimiate, int skipIndex, TIntDoubleHashMap rowToModify)
      {
         this.valueToElimiate = valueToElimiate;
         this.skipIndex = skipIndex;
         this.rowToModify = rowToModify;
      }

      @Override
      public boolean execute(int index, double value)
      {
         // Skip since this one is already removed.
         if (index == skipIndex)
         {
            return true;
         }

         double adjust = -valueToElimiate * value;
         rowToModify.adjustOrPutValue(index, adjust, adjust);
         return true;
      }
   }

   private class RowScalingProcedure implements TDoubleFunction
   {
      private double scale;

      public void set(double scale)
      {
         this.scale = scale;
      }

      @Override
      public double execute(double value)
      {
         return value * scale;
      }

   }
}
