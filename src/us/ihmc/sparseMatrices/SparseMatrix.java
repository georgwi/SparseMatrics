package us.ihmc.sparseMatrices;

import org.ejml.data.DenseMatrix64F;

import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.procedure.TIntDoubleProcedure;
import gnu.trove.procedure.TIntObjectProcedure;

public class SparseMatrix
{
   private static final int defaultInitialSize = 100;

   /**
    * A map from matrix row index to row. Each row is represented by a map from column index to the entry.
    */
   private final TIntObjectHashMap<TIntDoubleHashMap> values;

   /**
    * The number of rows in this matrix.
    */
   private int rows;

   /**
    * The number of columns in this matrix.
    */
   private int columns;

   public SparseMatrix()
   {
      this(0, 0);
   }

   public SparseMatrix(int rows, int colums)
   {
      values = createDataStructure(defaultInitialSize);
      resize(rows, colums);
   }

   private static TIntObjectHashMap<TIntDoubleHashMap> createDataStructure(int initialCapacity)
   {
      TIntObjectHashMap<TIntDoubleHashMap> values = new TIntObjectHashMap<>(initialCapacity);
      values.setAutoCompactionFactor(0f);
      for (int i = 0; i < defaultInitialSize; i++)
      {
         values.put(i, createRow(defaultInitialSize));
      }
      return values;
   }

   private static TIntDoubleHashMap createRow(int initialCapacity)
   {
      TIntDoubleHashMap row = new TIntDoubleHashMap(initialCapacity);
      row.setAutoCompactionFactor(0f);
      return row;
   }

   public void set(int rowIdx, int colIdx, double value)
   {
      checkDimentions(rowIdx, colIdx);
      if (value == 0.0)
      {
         return;
      }
      setUnsafe(rowIdx, colIdx, value);
   }

   private void setUnsafe(int rowIdx, int colIdx, double value)
   {
      TIntDoubleHashMap row = values.get(rowIdx);
      if (row == null)
      {
         row = createRow(defaultInitialSize);
         values.put(rowIdx, row);
      }
      row.put(colIdx, value);
   }

   public void setRow(int rowIdx, TIntDoubleHashMap row)
   {
      TIntDoubleHashMap localRow = values.get(rowIdx);
      if (localRow == null)
      {
         localRow = createRow(defaultInitialSize);
         values.put(rowIdx, localRow);
      }
      copy(localRow, row);
   }

   public void set(SparseMatrix matrix)
   {
      copy(this, matrix);
   }

   public void set(DenseMatrix64F denseMatrix, double epsilon)
   {
      resize(denseMatrix.numRows, denseMatrix.numCols);
      for (int row = 0; row < denseMatrix.numRows; row++)
      {
         for (int column = 0; column < denseMatrix.numCols; column++)
         {
            int value = denseMatrix.getIndex(row, column);
            if (value > epsilon || value < -epsilon)
            {
               setUnsafe(row, column, value);
            }
         }
      }
   }

   public void setIdentity(int size)
   {
      resize(size, size);
      for (int i = 0; i < size; i++)
      {
         TIntDoubleHashMap row = values.get(i);
         if (row == null)
         {
            row = createRow(defaultInitialSize);
            values.put(i, row);
         }
         else
         {
            row.clear();
         }
         row.put(i, 1.0);
      }
   }

   public void add(int rowIdx, int colIdx, double value)
   {
      checkDimentions(rowIdx, colIdx);
      if (value == 0.0)
      {
         return;
      }
      addUnsafe(rowIdx, colIdx, value);
   }

   private void addUnsafe(int rowIdx, int colIdx, double value)
   {
      TIntDoubleHashMap row = values.get(rowIdx);
      if (row == null)
      {
         row = createRow(defaultInitialSize);
         values.put(rowIdx, row);
         row.put(colIdx, value);
      }
      else if (!row.adjustValue(colIdx, value))
      {
         row.put(colIdx, value);
      }
   }

   public double get(int rowIdx, int colIdx)
   {
      checkDimentions(rowIdx, colIdx);
      TIntDoubleHashMap row = values.get(rowIdx);
      if (row == null)
      {
         return 0.0;
      }
      return row.get(colIdx);
   }

   public boolean contains(int rowIdx, int colIdx)
   {
      TIntDoubleHashMap row = values.get(rowIdx);
      if (row == null)
      {
         return false;
      }
      return row.contains(colIdx);
   }

   public void resize(int rows, int colums)
   {
      this.rows = rows;
      this.columns = colums;
   }

   public void clear()
   {
      values.forEachEntry(clearingProcedure);
   }

   public void mult(SparseMatrix matrixA, SparseMatrix matrixB)
   {
      if (matrixA.getColumns() != matrixB.getRows())
      {
         throw new RuntimeException("Unexpected Dimensions for Multiplication.");
      }

      SparseMatrix.multiply(matrixA, matrixB, this);
   }

   public int getRows()
   {
      return rows;
   }

   public int getColumns()
   {
      return columns;
   }

   public TIntDoubleHashMap getRow(int rowIdx)
   {
      if (rowIdx < 0 || rowIdx >= rows)
      {
         throw new RuntimeException("Unexpected index.");
      }
      return values.get(rowIdx);
   }

   private void checkDimentions(int rowIdx, int colIdx)
   {
      if (rowIdx < 0 || rowIdx >= rows || colIdx < 0 || colIdx >= columns)
      {
         throw new RuntimeException("Unexpected index.");
      }
   }

   @Override
   public String toString()
   {
      StringBuilder builder = new StringBuilder();
      builder.append("Matrix of size " + rows + "x" + columns);
      for (int rowIndex = 0; rowIndex < rows; rowIndex++)
      {
         if (rows == 1)
         {
            builder.append("\n <");
         }
         else if (rowIndex == 0)
         {
            builder.append("\n /");
         }
         else if (rowIndex == rows - 1)
         {
            builder.append("\n \\");
         }
         else
         {
            builder.append("\n |");
         }

         for (int columnIndex = 0; columnIndex < columns; columnIndex++)
         {
            if (contains(rowIndex, columnIndex))
            {
               double value = get(rowIndex, columnIndex);
               if (value >= 0.0)
               {
                  builder.append(" ");
               }
               builder.append(String.format("%.2f", value));
            }
            else
            {
               builder.append("  *  ");
            }
            if (columnIndex != columns - 1)
            {
               builder.append(", ");
            }
         }

         if (rows == 1)
         {
            builder.append(">");
         }
         else if (rowIndex == 0)
         {
            builder.append("\\");
         }
         else if (rowIndex == rows - 1)
         {
            builder.append("/");
         }
         else
         {
            builder.append("|");
         }
      }
      return builder.toString();
   }

   private static final ClearingProcedure clearingProcedure = new ClearingProcedure();
   private static class ClearingProcedure implements TIntObjectProcedure<TIntDoubleHashMap>
   {
      @Override
      public boolean execute(int key, TIntDoubleHashMap entry)
      {
         entry.clear();
         return true;
      }
   }

   /**
    * Set c = a * b.
    */
   private static void multiply(SparseMatrix a, SparseMatrix b, SparseMatrix c)
   {
      c.resize(a.getRows(), b.getColumns());
      c.clear();
      multProcedureA.set(c, b);
      a.values.forEachEntry(multProcedureA);
   }

   private static final MultProcedureA multProcedureA = new MultProcedureA();
   private static class MultProcedureA implements TIntObjectProcedure<TIntDoubleHashMap>
   {
      private SparseMatrix matrix;
      private SparseMatrix B;

      public void set(SparseMatrix matrix, SparseMatrix B)
      {
         this.matrix = matrix;
         this.B = B;
      }

      @Override
      public boolean execute(int rowIndexA, TIntDoubleHashMap rowA)
      {
         multProcedureB.set(rowIndexA, matrix, B);
         rowA.forEachEntry(multProcedureB);
         return true;
      }
   }

   private static final MultProcedureB multProcedureB = new MultProcedureB();
   private static class MultProcedureB implements TIntDoubleProcedure
   {
      private int rowIndexA;
      private SparseMatrix matrix;
      private SparseMatrix B;

      public void set(int rowIndexA, SparseMatrix matrix, SparseMatrix B)
      {
         this.rowIndexA = rowIndexA;
         this.matrix = matrix;
         this.B = B;
      }

      @Override
      public boolean execute(int colIndexA, double valueA)
      {
         int rowIndexB = colIndexA;
         TIntDoubleHashMap rowB = B.values.get(rowIndexB);

         if (rowB != null && !rowB.isEmpty())
         {
            multProcedureC.set(rowIndexA, valueA, matrix);
            rowB.forEachEntry(multProcedureC);
         }

         return true;
      }
   }

   private static final MultProcedureC multProcedureC = new MultProcedureC();
   private static class MultProcedureC implements TIntDoubleProcedure
   {
      private int rowIndexA;
      private double valueA;
      private SparseMatrix matrix;

      public void set(int rowIndexA, double valueA, SparseMatrix matrix)
      {
         this.rowIndexA = rowIndexA;
         this.valueA = valueA;
         this.matrix = matrix;
      }

      @Override
      public boolean execute(int colIndexB, double valueB)
      {
         matrix.add(rowIndexA, colIndexB, valueA * valueB);
         return true;
      }
   }

   /**
    * Set a = b.
    */
   private static void copy(SparseMatrix a, SparseMatrix b)
   {
      a.resize(b.getRows(), b.getColumns());
      a.clear();
      copyProcedureA.set(a);
      b.values.forEachEntry(copyProcedureA);
   }

   private static final CopyProcedureA copyProcedureA = new CopyProcedureA();
   private static class CopyProcedureA implements TIntObjectProcedure<TIntDoubleHashMap>
   {
      private SparseMatrix matrixToSet;

      public void set(SparseMatrix matrixToSet)
      {
         this.matrixToSet = matrixToSet;
      }

      @Override
      public boolean execute(int rowIdx, TIntDoubleHashMap row)
      {
         copyProcedureB.set(rowIdx, matrixToSet);
         row.forEachEntry(copyProcedureB);
         return true;
      }
   }

   private static final CopyProcedureB copyProcedureB = new CopyProcedureB();
   private static class CopyProcedureB implements TIntDoubleProcedure
   {
      private int rowIdx;
      private SparseMatrix matrixToSet;

      public void set(int rowIdx, SparseMatrix matrixToSet)
      {
         this.rowIdx = rowIdx;
         this.matrixToSet = matrixToSet;
      }

      @Override
      public boolean execute(int columnIdx, double value)
      {
         matrixToSet.setUnsafe(rowIdx, columnIdx, value);
         return true;
      }
   }

   /**
    * Sets a = b.
    */
   private static void copy(TIntDoubleHashMap a, TIntDoubleHashMap b)
   {
      a.clear();
      a.putAll(b);
   }
}
