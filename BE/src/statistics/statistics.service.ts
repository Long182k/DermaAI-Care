import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma.service';
import { Role, Gender, ScheduleStatus } from '@prisma/client';
import * as moment from 'moment';

@Injectable()
export class StatisticsService {
  constructor(private prisma: PrismaService) {}

  // Modify the getOverview method to use sequential queries instead of Promise.all
  // Add caching properties
  private overviewCache: any = null;
  private overviewCacheTime: Date | null = null;
  private readonly CACHE_TTL = 5 * 60 * 1000; // 5 minutes in milliseconds

  async getOverview() {
    // Check if cache is valid
    if (
      this.overviewCache &&
      this.overviewCacheTime &&
      new Date().getTime() - this.overviewCacheTime.getTime() < this.CACHE_TTL
    ) {
      return this.overviewCache;
    }

    // If no valid cache, fetch from database
    // Get counts sequentially to avoid opening too many connections
    const totalPatients = await this.prisma.user.count({
      where: { role: Role.PATIENT },
    });
    const totalDoctors = await this.prisma.user.count({
      where: { role: Role.DOCTOR },
    });
    const totalPredictions = await this.prisma.prediction.count();
    const totalAppointments = await this.prisma.appointment.count();
    const totalPayments = await this.prisma.payment.count();

    // Get monthly data for charts
    const currentYear = new Date().getFullYear();
    const monthlyStats = await this.getMonthlyStats(currentYear);

    const result = {
      summary: {
        totalPatients,
        totalDoctors,
        totalPredictions,
        totalAppointments,
        totalPayments,
      },
      charts: {
        monthly: monthlyStats,
      },
    };

    // Store in cache
    this.overviewCache = result;
    this.overviewCacheTime = new Date();

    return result;
  }

  // Also modify getMonthlyStats to use sequential queries
  private async getMonthlyStats(year: number) {
    const startOfYear = new Date(year, 0, 1);
    const endOfYear = new Date(year, 11, 31, 23, 59, 59);

    // Get data sequentially instead of in parallel
    const patients = await this.prisma.user.findMany({
      where: {
        role: Role.PATIENT,
        createdAt: {
          gte: startOfYear,
          lte: endOfYear,
        },
      },
      select: { createdAt: true },
    });

    const doctors = await this.prisma.user.findMany({
      where: {
        role: Role.DOCTOR,
        createdAt: {
          gte: startOfYear,
          lte: endOfYear,
        },
      },
      select: { createdAt: true },
    });

    const predictions = await this.prisma.prediction.findMany({
      where: {
        createdAt: {
          gte: startOfYear,
          lte: endOfYear,
        },
      },
      select: { createdAt: true },
    });

    const appointments = await this.prisma.appointment.findMany({
      where: {
        createdAt: {
          gte: startOfYear,
          lte: endOfYear,
        },
      },
      select: { createdAt: true },
    });

    // Initialize monthly counters
    const monthlyPatients = Array(12).fill(0);
    const monthlyDoctors = Array(12).fill(0);
    const monthlyPredictions = Array(12).fill(0);
    const monthlyAppointments = Array(12).fill(0);

    // Count items by month
    patients.forEach((item) => {
      const month = new Date(item.createdAt).getMonth();
      monthlyPatients[month]++;
    });

    doctors.forEach((item) => {
      const month = new Date(item.createdAt).getMonth();
      monthlyDoctors[month]++;
    });

    predictions.forEach((item) => {
      const month = new Date(item.createdAt).getMonth();
      monthlyPredictions[month]++;
    });

    appointments.forEach((item) => {
      const month = new Date(item.createdAt).getMonth();
      monthlyAppointments[month]++;
    });

    return {
      labels: [
        'Jan',
        'Feb',
        'Mar',
        'Apr',
        'May',
        'Jun',
        'Jul',
        'Aug',
        'Sep',
        'Oct',
        'Nov',
        'Dec',
      ],
      datasets: [
        { label: 'Patients', data: monthlyPatients },
        { label: 'Doctors', data: monthlyDoctors },
        { label: 'Predictions', data: monthlyPredictions },
        { label: 'Appointments', data: monthlyAppointments },
      ],
    };
  }

  async getPatients() {
    // Get basic patient statistics
    const totalPatients = await this.prisma.user.count({
      where: { role: Role.PATIENT },
    });

    // Get gender distribution
    const genderDistribution = await this.prisma.user.groupBy({
      by: ['gender'],
      where: { role: Role.PATIENT },
      _count: { id: true },
    });

    // Get patients by registration date (monthly for the current year)
    const currentYear = new Date().getFullYear();
    const startOfYear = new Date(currentYear, 0, 1);

    const patientsByMonth = await this.prisma.user.findMany({
      where: {
        role: Role.PATIENT,
        createdAt: { gte: startOfYear },
      },
      select: {
        id: true,
        createdAt: true,
        gender: true,
      },
    });

    // Process data for monthly chart
    const monthlyData = Array(12).fill(0);
    patientsByMonth.forEach((patient) => {
      const month = new Date(patient.createdAt).getMonth();
      monthlyData[month]++;
    });

    // Get patients with most appointments
    const patientsWithAppointments = await this.prisma.user.findMany({
      where: { role: Role.PATIENT },
      select: {
        id: true,
        userName: true,
        firstName: true,
        lastName: true,
        isActive: true,
        _count: {
          select: { PatientAppointments: true },
        },
      },
      orderBy: {
        PatientAppointments: { _count: 'desc' },
      },
      take: 10,
    });

    // Get patients with most predictions
    const patientsWithPredictions = await this.prisma.user.findMany({
      where: { role: Role.PATIENT },
      select: {
        id: true,
        userName: true,
        firstName: true,
        lastName: true,
        isActive: true,
        _count: {
          select: { Predictions: true },
        },
      },
      orderBy: {
        Predictions: { _count: 'desc' },
      },
      take: 10,
    });

    return {
      summary: {
        totalPatients,
        genderDistribution: this.formatGenderDistribution(genderDistribution),
      },
      charts: {
        monthlyRegistrations: {
          labels: [
            'Jan',
            'Feb',
            'Mar',
            'Apr',
            'May',
            'Jun',
            'Jul',
            'Aug',
            'Sep',
            'Oct',
            'Nov',
            'Dec',
          ],
          data: monthlyData,
        },
        topPatientsByAppointments: patientsWithAppointments.map((p) => ({
          id: p.id,
          isActive: p.isActive,
          name: this.formatUserName(p),
          count: p._count.PatientAppointments,
        })),
        topPatientsByPredictions: patientsWithPredictions.map((p) => ({
          id: p.id,
          isActive: p.isActive,
          name: this.formatUserName(p),
          count: p._count.Predictions,
        })),
      },
    };
  }

  async getDoctors() {
    // Get basic doctor statistics
    const totalDoctors = await this.prisma.user.count({
      where: { role: Role.DOCTOR },
    });

    // Get gender distribution
    const genderDistribution = await this.prisma.user.groupBy({
      by: ['gender'],
      where: { role: Role.DOCTOR },
      _count: { id: true },
    });

    // Get doctors with most appointments
    const doctorsWithAppointments = await this.prisma.user.findMany({
      where: { role: Role.DOCTOR },
      select: {
        id: true,
        userName: true,
        firstName: true,
        lastName: true,
        isActive: true,
        _count: {
          select: { DoctorAppointments: true },
        },
      },
      orderBy: {
        DoctorAppointments: { _count: 'desc' },
      },
      take: 10,
    });

    // Get doctors by experience
    const doctorsByExperience = await this.prisma.user.groupBy({
      by: ['experience'],
      where: {
        role: Role.DOCTOR,
        experience: { not: null },
      },
      _count: { id: true },
    });

    // Get doctors by languages
    const doctorsByLanguages = await this.prisma.user.groupBy({
      by: ['languages'],
      where: { role: Role.DOCTOR },
      _count: { id: true },
    });

    return {
      summary: {
        totalDoctors,
        genderDistribution: this.formatGenderDistribution(genderDistribution),
      },
      charts: {
        topDoctorsByAppointments: doctorsWithAppointments.map((d) => ({
          id: d.id,
          isActive: d.isActive,
          name: this.formatUserName(d),
          count: d._count.DoctorAppointments,
        })),
        doctorsByExperience: doctorsByExperience
          .filter((item) => item.experience !== null)
          .sort((a, b) => (a.experience || 0) - (b.experience || 0))
          .map((item) => ({
            experience: item.experience || 0,
            count: item._count.id,
          })),
        doctorsByLanguages: doctorsByLanguages.map((item) => ({
          language: item.languages || 'Not Specified',
          count: item._count.id,
        })),
      },
    };
  }

  async getPredictions() {
    // Define the lesion classes of interest
    const CLASS_MAP = {
      MEL: 'Melanoma',
      NV: 'Nevus',
      BCC: 'Basal Cell Carcinoma',
      AK: 'Actinic Keratosis',
      BKL: 'Benign Keratosis',
      DF: 'Dermatofibroma',
      VASC: 'Vascular Lesion',
      SCC: 'Squamous Cell Carcinoma',
    };
    const classKeys = Object.keys(CLASS_MAP);

    // Fetch all predictions (limit if needed for performance)
    const predictions = await this.prisma.prediction.findMany({
      select: { result: true, createdAt: true },
    });

    // Count detections by class
    const classCounts: Record<string, number> = {};
    let totalDetections = 0;

    for (const prediction of predictions) {
      try {
        const result = prediction.result;
        if (
          result &&
          typeof result === 'object' &&
          !Array.isArray(result) &&
          Array.isArray((result as any).detections)
        ) {
          for (const detection of (result as any).detections) {
            const lesionClass = detection.class;
            if (classKeys.includes(lesionClass)) {
              classCounts[lesionClass] = (classCounts[lesionClass] || 0) + 1;
              totalDetections++;
            }
          }
        }
      } catch (err) {
        // Ignore parse errors
      }
    }

    // Prepare data for chart: array of { class, name, count, percent }
    const lesionDistribution = classKeys.map((key) => {
      // Find the most recent detection with explanation for this class
      let isCancerous: boolean | null = null;
      for (let i = predictions.length - 1; i >= 0; i--) {
        const result = predictions[i].result;
        if (
          result &&
          typeof result === 'object' &&
          !Array.isArray(result) &&
          Array.isArray((result as any).detections)
        ) {
          for (const detection of (result as any).detections) {
            if (detection.class === key && detection.explanation) {
              isCancerous = detection.explanation.isCancerous;
              break;
            }
          }
        }
        if (isCancerous !== null) break;
      }
      return {
        class: key,
        name: CLASS_MAP[key],
        count: classCounts[key] || 0,
        percent:
          totalDetections > 0
            ? Math.round(((classCounts[key] || 0) / totalDetections) * 100)
            : 0,
        isCancerous,
      };
    });

    // Prepare monthly predictions data
    const monthlyCounts = Array(12).fill(0);
    for (const prediction of predictions) {
      const createdAt = prediction.createdAt
        ? new Date(prediction.createdAt)
        : null;
      if (createdAt) {
        const month = createdAt.getMonth();
        monthlyCounts[month]++;
      }
    }

    return {
      summary: {
        totalPredictions: predictions.length,
        lesionDistribution,
      },
      charts: {
        lesionDistribution,
        monthlyPredictions: {
          labels: [
            'Jan',
            'Feb',
            'Mar',
            'Apr',
            'May',
            'Jun',
            'Jul',
            'Aug',
            'Sep',
            'Oct',
            'Nov',
            'Dec',
          ],
          data: monthlyCounts,
        },
      },
    };
  }

  async getAppointments() {
    // Get basic appointment statistics
    const totalAppointments = await this.prisma.appointment.count();

    // Get appointments by status
    const appointmentsByStatus = await this.prisma.appointment.groupBy({
      by: ['status'],
      _count: { id: true },
    });

    // Get appointments by month (for current year)
    const currentYear = new Date().getFullYear();
    const startOfYear = new Date(currentYear, 0, 1);

    const appointmentsByMonth = await this.prisma.appointment.findMany({
      where: { createdAt: { gte: startOfYear } },
      select: {
        id: true,
        createdAt: true,
        status: true,
      },
    });

    // Process data for monthly chart
    const monthlyData = Array(12).fill(0);
    const monthlyByStatus = {};

    appointmentsByMonth.forEach((appointment) => {
      const month = new Date(appointment.createdAt).getMonth();
      monthlyData[month]++;

      // Track by status
      if (!monthlyByStatus[appointment.status]) {
        monthlyByStatus[appointment.status] = Array(12).fill(0);
      }
      monthlyByStatus[appointment.status][month]++;
    });

    // Get upcoming appointments (next 7 days)
    const today = new Date();
    const nextWeek = new Date(today);
    nextWeek.setDate(today.getDate() + 7);

    const upcomingAppointments = await this.prisma.appointment.findMany({
      where: {
        Schedule: {
          startTime: {
            gte: today,
            lte: nextWeek,
          },
        },
        status: 'SCHEDULED',
      },
      include: {
        Patient: {
          select: {
            userName: true,
            firstName: true,
            lastName: true,
          },
        },
        Doctor: {
          select: {
            userName: true,
            firstName: true,
            lastName: true,
          },
        },
        Schedule: true,
      },
      orderBy: {
        Schedule: {
          startTime: 'asc',
        },
      },
    });

    // Get appointment completion rate
    const completedAppointments = await this.prisma.appointment.count({
      where: { status: 'COMPLETED' },
    });

    const completionRate =
      totalAppointments > 0
        ? (completedAppointments / totalAppointments) * 100
        : 0;

    const confirmedAppointments = await this.prisma.appointment.findMany({
      where: { status: 'CONFIRMED' },
      select: {
        id: true,
        createdAt: true,
        Schedule: {
          select: { startTime: true },
        },
      },
    });
    const daysOfWeek = [
      'Sunday',
      'Monday',
      'Tuesday',
      'Wednesday',
      'Thursday',
      'Friday',
      'Saturday',
    ];
    const appointmentsByDay = daysOfWeek.map((day) => ({ day, count: 0 }));
    confirmedAppointments.forEach((appointment) => {
      const date = appointment.Schedule?.startTime
        ? new Date(appointment.Schedule.startTime)
        : appointment.createdAt
          ? new Date(appointment.createdAt)
          : null;
      if (date) {
        const dayIndex = date.getDay();
        appointmentsByDay[dayIndex].count += 1;
      }
    });

    // Get all appointments with required fields
    const allAppointments = await this.prisma.appointment.findMany({
      include: {
        Patient: {
          select: {
            userName: true,
            firstName: true,
            lastName: true,
          },
        },
        Doctor: {
          select: {
            userName: true,
            firstName: true,
            lastName: true,
          },
        },
        Schedule: true,
      },
      orderBy: {
        createdAt: 'desc',
      },
    });

    // Format the appointments list with required fields
    const appointmentsList = allAppointments.map((appointment) => ({
      id: appointment.id,
      patientName: this.formatUserName(appointment.Patient),
      doctorName: this.formatUserName(appointment.Doctor),
      status: appointment.status,
      notes: appointment.notes,
      startTime: appointment.Schedule.startTime,
      endTime: appointment.Schedule.endTime,
    }));

    return {
      summary: {
        totalAppointments,
        completedAppointments,
        completionRate: Math.round(completionRate * 100) / 100, // Round to 2 decimal places
        appointmentsByStatus: appointmentsByStatus.map((item) => ({
          status: item.status,
          count: item._count.id,
        })),
      },
      charts: {
        monthlyAppointments: {
          labels: [
            'Jan',
            'Feb',
            'Mar',
            'Apr',
            'May',
            'Jun',
            'Jul',
            'Aug',
            'Sep',
            'Oct',
            'Nov',
            'Dec',
          ],
          data: monthlyData,
        },
        monthlyAppointmentsByStatus: Object.entries(monthlyByStatus).map(
          ([status, data]) => ({
            status,
            data,
          }),
        ),
        upcomingAppointments: upcomingAppointments.map((appointment) => ({
          id: appointment.id,
          patientName: this.formatUserName(appointment.Patient),
          doctorName: this.formatUserName(appointment.Doctor),
          startTime: appointment.Schedule.startTime,
          endTime: appointment.Schedule.endTime,
          status: appointment.status,
        })),
        appointmentsByDay,
      },
      appointmentsList, // Added the list of all appointments with required fields
    };
  }

  async getPaymentStatistics() {
    // Get all payments
    const payments = await this.prisma.payment.findMany();

    // Total payments
    const totalPayments = payments.length;

    // Total completed payments
    const completedPayments = payments.filter(
      (p) => p.status === 'COMPLETED',
    ).length;

    // Total pending payments
    const pendingPayments = payments.filter(
      (p) => p.status === 'PENDING',
    ).length;

    // Total amount (completed)
    const totalAmountCompleted = payments
      .filter((p) => p.status === 'COMPLETED')
      .reduce((sum, p) => sum + Number(p.amount), 0);

    // Payments by status
    const paymentsByStatus = ['COMPLETED', 'PENDING', 'FAILED'].map(
      (status) => ({
        status,
        count: payments.filter((p) => p.status === status).length,
        totalAmount: payments
          .filter((p) => p.status === status)
          .reduce((sum, p) => sum + Number(p.amount), 0),
      }),
    );

    // Payments by day (for chart)
    const daysOfWeek = [
      'Sunday',
      'Monday',
      'Tuesday',
      'Wednesday',
      'Thursday',
      'Friday',
      'Saturday',
    ];
    const paymentsByDay = daysOfWeek.map((day) => ({
      day,
      count: 0,
      totalAmount: 0,
    }));
    payments.forEach((payment) => {
      const date = new Date(payment.createdAt);
      const dayIndex = date.getDay();
      paymentsByDay[dayIndex].count += 1;
      paymentsByDay[dayIndex].totalAmount += Number(payment.amount);
    });

    // Payments by month (for chart)
    const monthlyPayments = Array(12).fill(0);
    payments.forEach((payment) => {
      const date = new Date(payment.createdAt);
      const month = date.getMonth();
      monthlyPayments[month] += Number(payment.amount);
    });

    return {
      summary: {
        totalPayments,
        completedPayments,
        pendingPayments,
        totalAmountCompleted,
        paymentsByStatus,
      },
      charts: {
        paymentsByDay,
        monthlyPayments: {
          labels: [
            'Jan',
            'Feb',
            'Mar',
            'Apr',
            'May',
            'Jun',
            'Jul',
            'Aug',
            'Sep',
            'Oct',
            'Nov',
            'Dec',
          ],
          data: monthlyPayments,
        },
      },
    };
  }

  private formatGenderDistribution(genderData: any[]) {
    const result = {
      MALE: 0,
      FEMALE: 0,
      OTHER: 0,
    };

    genderData.forEach((item) => {
      if (item.gender) {
        result[item.gender] = item._count.id;
      } else {
        result['OTHER'] += item._count.id;
      }
    });

    return Object.entries(result).map(([gender, count]) => ({
      gender,
      count,
    }));
  }

  private formatUserName(user: any) {
    if (user.firstName && user.lastName) {
      return `${user.firstName} ${user.lastName}`;
    }
    return user.userName;
  }
}
