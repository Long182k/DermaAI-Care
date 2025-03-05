import {
  Body,
  Controller,
  Get,
  Param,
  Patch,
  Post,
  Query,
  UseGuards,
} from '@nestjs/common';
import { User } from '@prisma/client';
import { CurrentUser } from '../auth/@decorator/current-user.decorator';
import { JwtAuthGuard } from '../auth/@guard/jwt-auth.guard';
import { RolesGuard } from '../auth/@guard/roles.guard';
import { Roles } from '../auth/@decorator/roles.decorator';
import { UpdateScheduleDto } from './dto/update-schedule.dto';
import { ScheduleService } from './schedule.service';
import { ScheduleTasksService } from './schedule-tasks.service';
import { ROLE } from 'src/auth/util/@enum/role.enum';
import { ParseISODatePipe } from '../common/pipes/parse-iso-date.pipe';

@Controller('schedules')
@UseGuards(JwtAuthGuard, RolesGuard)
export class ScheduleController {
  constructor(
    private readonly scheduleService: ScheduleService,
    private readonly scheduleTasksService: ScheduleTasksService,
  ) {}

  @Get('doctor/:doctorId')
  async getDoctorSchedules(
    @Param('doctorId') doctorId: string,
    @Query('startDate', ParseISODatePipe) startDate: Date,
    @Query('endDate', ParseISODatePipe) endDate: Date,
  ) {
    return await this.scheduleService.findDoctorSchedules(
      doctorId,
      startDate,
      endDate,
    );
  }

  @Patch(':id')
  async updateSchedule(
    @Param('id') id: string,
    @Body() updateScheduleDto: UpdateScheduleDto,
    @CurrentUser() currentUser: User,
  ) {
    return await this.scheduleService.updateSchedule(
      id,
      updateScheduleDto,
      currentUser,
    );
  }

  @Post('generate')
  // @Roles(ROLE.ADMIN)
  async generateSchedules() {
    return await this.scheduleTasksService.generateSchedulesForDoctors();
  }
}
